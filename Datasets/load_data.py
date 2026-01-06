import json
from pathlib import Path

def process_data(dataset_path):
    # ---------------- Load dataset ----------------
    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    dataset = []
    # prompts, labels, categories = [], [], []
    for i, ex in enumerate(examples):
        sample = {
            "id": i,
        }

        question = ex["instruction"].strip()
        options = ex.get("input", "").strip()       # 若有input字段则取，若无则取"" -> 空值
        label = str(ex["output"]).strip().upper()   # 选择题
        category = ex.get("category") if "category" in ex else None

        if options: # options有的话，就跟question链接。
            prompt = f"{question}\n\n{options}\n\nAnswer:"
        else:       # 无的话选项就在question里
            prompt = f"{question}\n\nAnswer:"

        sample["prompt"]   = prompt
        sample["label"]    = label
        sample["category"] = category   # Only BBH has category

        dataset.append(sample)
    
    return dataset

def process_ocr_data(CLIP=10**10):
    """
    OmniDocBench: https://github.com/opendatalab/OmniDocBench.git
    """
    OmniDir = Path("/mnt/zhangchen/S3Precision/LLM-inference/Datasets/OmniDocBench/")
    omni_images_dir = (OmniDir / "demo_data" / "omnidocbench_demo" / "images").resolve()
    omni_pred_dir   = (OmniDir / "demo_data" / "end2end").resolve()

    datasets = []
    for index, file in enumerate(omni_images_dir.iterdir()):
        sample = {
            "id": index
        }
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:    # suffix -> 返回文件后缀名
            sample["img_path"] = file
            datasets.append(sample)
    
    sorted(datasets, key=lambda x: x["img_path"].suffix)
    return datasets[0:CLIP]


if __name__ == "__main__":
    Dir = "/mnt/zhangchen/S3Precision/LLM-inference/Datasets/"
    dataset = process_data(Dir + "BBH/test.json")
    print(dataset[0], len(dataset))
    dataset = process_data(Dir + "GPQA-diamond/test.json")
    print(dataset[0], len(dataset))
    dataset = process_data(Dir + "MMLU-PRO/test.json")
    print(dataset[0], len(dataset))
