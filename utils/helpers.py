import os
import json
import torch
import random

def same_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def getContent(arch, precision):
    from contextlib import nullcontext
    # transformer engine
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    content = nullcontext()
    if arch == "NV":
        if precision == "mxfp8":
            te_recipe = recipe.MXFP8BlockScaling(fp8_format=recipe.Format.HYBRID)
        elif precision == "nvfp4":
            te_recipe = recipe.NVFP4BlockScaling()
        else:
            te_recipe = None
        content = te.autocast(enabled=True, recipe=te_recipe)
    return content

def save_data_to_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
