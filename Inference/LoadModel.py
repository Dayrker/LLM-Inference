import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Model Config
def getModel(model_dir, cuda_maps="cuda:0", dtype=torch.bfloat16):
    # print("model_dir:", model_dir)
    # 4.53.1 -> torch_dtype, 4.57.3 -> dtype
    
    if "DeepSeek-OCR" in model_dir:
        model_ori = AutoModel.from_pretrained(
            model_dir,
            device_map=cuda_maps,  
            torch_dtype=dtype,     
            use_safetensors=True,
            trust_remote_code=True,
            # _attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            # padding_side='left',    # !! Important
            trust_remote_code=True,
            use_fast=True
        )
    else:
        model_ori = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=cuda_maps,  
            torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            padding_side='left',    # !! Important
            trust_remote_code=True,
            use_fast=True
        )
    model_ori.config.use_cache = False
    model_ori.config.pretraining_tp = 1
    
    
    if "llama" in model_dir:   # Llama needs supplement pad_token
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model_ori, tokenizer
