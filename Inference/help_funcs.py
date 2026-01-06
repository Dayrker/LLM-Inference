
import torch
import random
import io
import sys
import contextlib

def same_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def getContent(arch, precision):
    from contextlib import nullcontext
    # transformer engine
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    # content = nullcontext()
    
    if precision == "mxfp8":
        te_recipe = recipe.MXFP8BlockScaling(fp8_format=recipe.Format.HYBRID)
    elif precision == "nvfp4":
        te_recipe = recipe.NVFP4BlockScaling()
    else:
        te_recipe = None
    content = te.autocast(enabled=True, recipe=te_recipe)
    return content


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to silence stdout from noisy libraries."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
