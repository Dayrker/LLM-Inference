import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model Config
def getModel(model_dir, cuda_maps="cuda:0", dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, 
        padding_side='left',    # !! Important
        use_fast=True)
    model_ori = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=cuda_maps,  
        torch_dtype=dtype,     # 4.53.1 -> torch_dtype, 4.57.3 -> dtype
    )
    model_ori.config.use_cache = False
    model_ori.config.pretraining_tp = 1
    
    
    if "llama" in model_dir:   # Llama needs supplement pad_token
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model_ori, tokenizer
