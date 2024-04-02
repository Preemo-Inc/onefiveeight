import argparse

from datasets import load_dataset
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TheBloke/Llama-2-13B-fp16")
    parser.add_argument("--use_quant", type=bool, default=True)
    parser.add_argument("--group_size", type=int, default=16)

    args = parser.parse_args()
    print(f"Running evaluation with {args=}")

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

        quant_config = BaseQuantizeConfig(nbits=1.58, group_size=args.group_size)
        model.quantize_model(
            quant_config=quant_config, compute_dtype=compute_dtype, device=device
        )
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(model_id).to(torch.float16).to(device)
        )

    model.eval()


from examples.llama2_benchmark import eval_model

res = eval_model.eval_wikitext2(model, tokenizer)
print(res)
