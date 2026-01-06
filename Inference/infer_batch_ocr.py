import torch
from tqdm import tqdm
from typing import List, Literal, Optional
from pathlib import Path
# Parallel
import torch.multiprocessing as mp
# process model
from .LoadModel import getModel 
from .convert_model import replace_modules
from .help_funcs import same_seed, getContent, suppress_stdout


RESOLUTIONS = {
    "tiny":   dict(base_size=512,  image_size=512,  crop_mode=False),
    "small":  dict(base_size=640,  image_size=640,  crop_mode=False),
    "base":   dict(base_size=1024, image_size=1024, crop_mode=False),
    "large":  dict(base_size=1280, image_size=1280, crop_mode=False),
    "gundam": dict(base_size=1024, image_size=640,  crop_mode=True),
}

def build_prompt(
    task: Literal["markdown", "plain", "layout_ocr", "describe", "figure", "custom"] = "markdown",
    custom_text: Optional[str] = None,
) -> str:
    if task == "custom" and custom_text:
        return f"<image>\n{custom_text}"

    mapping = {
        "markdown":   "<|grounding|>Convert the document to markdown.",
        "plain":      "Free OCR.",
        "layout_ocr": "<|grounding|>OCR this image.",
        "describe":   "Describe this image in detail.",
        "figure":     "Parse the figure.",
    }
    text = mapping.get(task, "Free OCR.")   # default: "Free OCR."
    return f"<image>\n{text}"


def infer_batch_ocr(args, dataset,
                    device = "cuda:0", return_queue = None):
    # Get Parameters
    arch      = args.arch
    precision = args.precision
    model     = args.model
    # Process model
    torch.cuda.set_device(device)   # 必须在此设置，不然会有illeagel memory
    ocr_model, tokenizer = getModel("/ssd/models/" + model, cuda_maps=device)  # 在mp.Process子进程中load model，速度才不会变慢，应该是内部有定义cuda
    replace_modules(ocr_model, arch, precision)
    # print("ocr_model TE:", ocr_model)

    # Get config
    prompt  = build_prompt("markdown")
    mode    = "large"
    res_cfg = RESOLUTIONS[mode]
    content = getContent(arch, precision)

    save_dir = Path("./results/OCR") / model / arch / precision
    for data in tqdm(dataset, desc="Benchmarking", unit="image"):
        img_path = data["img_path"]
        sub_dir  = save_dir / img_path.stem    # stem -> 文件名
        mmd_file = sub_dir  / "result.mmd"
        md_file  = sub_dir  / "result.md"

        # with suppress_stdout(), content: # Suppress infer output
        with content: # Suppress infer output
            result = ocr_model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(img_path),
                output_path=str(sub_dir),
                base_size=res_cfg["base_size"],
                image_size=res_cfg["image_size"],
                crop_mode=res_cfg["crop_mode"],
                save_results=True,
                test_compress=False,
            )
        # 结果写成.md文件
        md_content = mmd_file.read_text(encoding="utf-8").strip()
        md_file.write_text(md_content, encoding="utf-8")
