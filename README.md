# All things 1.58 Bit
## Roadmap
1. Check if we can pretrain from scratchwith 1.58 Bit (random initialized) (we are here)
2. Initialize 1.58 Bit from Mixtral/Mistral Weights (we are here)
3. Continued pretraining 
4. Move to ASIC
5. AGI (in 1.58 bit, on ASIC)

# Setup
```bash
python3 -m venv venv
. ./venv/bin/activate
cd hqq && pip install -e .
```

# TODO for 1Bit
- [x] create simple unit test
- [ ] check impl vs: https://huggingface.co/1bitLLM/bitnet_b1_58-xl/blob/main/utils_quant.py
- [ ] Check with https://x.com/NousResearch/status/1773923241268003052?s=20
- [ ] Check with https://huggingface.co/NousResearch/OLMo-Bitnet-1B


# HQQ -> 1.58
- [ ] test HQQ -> fork -> 1.58bit 
- [ ] 2bit quant llama / bitsandbytes

# Eval Results 

| Model                     | Dataset                                                  | Quant    | Groupsize | PPL    |
|---------------------------|----------------------------------------------------------|----------|-----------|--------|
| TheBloke/Llama-2-7B-fp16  | wikitext + wikitext_wikitext-2-raw-v1, validation splits | HQQ 1.58 | 16        | 400.46 |
| TheBloke/Llama-2-7B-fp16  | wikitext + wikitext_wikitext-2-raw-v1, validation splits | HQQ 1.58 | 8         | 8.69   |
| TheBloke/Llama-2-7B-fp16  | wikitext + wikitext_wikitext-2-raw-v1, validation splits | FP16     | -         | 5.18   |
| TheBloke/Llama-2-13B-fp16 | wikitext + wikitext_wikitext-2-raw-v1, validation splits | HQQ 1.58 | 16        | 48.23  |
| TheBloke/Llama-2-13B-fp16 | wikitext + wikitext_wikitext-2-raw-v1, validation splits | HQQ 1.58 | 8         | 7.2732 |

