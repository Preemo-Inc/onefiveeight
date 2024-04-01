import argparse

from datasets import load_dataset, Dataset
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer


# Prediction/Eval
######################################################################################
# from #https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
def compute_perplexity_batched(
    model,
    tokenizer,
    predictions,
    encodings=None,
    batch_size=1,
    add_start_token=True,
    device="cuda",
    max_length=None,
):
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    if encodings is None:
        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return np.mean(ppls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TheBloke/Llama-2-7B-fp16")
    parser.add_argument("--use_quant", type=bool, required=True)

    args = parser.parse_args()

    # Load the data
    wikitext_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")[
        "text"
    ]
    wikitext_val = [s for s in wikitext_val if s != ""]

    # Model and setttings
    model_id = args.model
    compute_dtype = torch.bfloat16
    device = "cuda:0"
    max_tokens = 256

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_quant:
        # Eval HQQ model
        # Load model on the CPU
        ######################
        model = HQQModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype)

        # Quantize the model
        ######################
        from hqq.core.quantize import *

        quant_config = BaseQuantizeConfig(nbits=1.58, group_size=16)
        model.quantize_model(
            quant_config=quant_config, compute_dtype=compute_dtype, device=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    model.eval()

print(
    f"{'HQQ 1.58' if args.use_quant else 'FP16':20s} perplexity:",
    compute_perplexity_batched(
        model=model,
        tokenizer=tokenizer,
        predictions=wikitext_val,
        batch_size=1,
        max_length=max_tokens,
        device=device,
    ),
)  # Prints "HQQ 1.58 perplexity 444.4851"
