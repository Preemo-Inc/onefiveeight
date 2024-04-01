from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
import torch 
#Model and setttings
model_id      = 'TheBloke/Llama-2-7B-fp16' # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
compute_dtype = torch.bfloat16
device        = 'cuda:0'

#Load model on the CPU
######################
model     = HQQModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id) 

#Quantize the model
######################
from hqq.core.quantize import *
quant_config = BaseQuantizeConfig(nbits=1.58, group_size=16)
model.quantize_model(quant_config=quant_config, compute_dtype=compute_dtype, device=device) 

# generate text

text = "Paris is the capital of"
input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
output = model.generate(input_ids, max_length=100)
string_out = tokenizer.decode(output[0], skip_special_tokens=True)
print(string_out)