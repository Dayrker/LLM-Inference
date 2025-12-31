import json

def process_data(dataset_path):
    # ---------------- Load dataset ----------------
    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    dataset = []
    # prompts, labels, categories = [], [], []
    for i, ex in enumerate(examples):
        sample = {
            "id:": i,
        }

        question = ex["instruction"].strip()
        options = ex.get("input", "").strip()       # 若有input字段则取，若无则取"" -> 空值
        label = str(ex["output"]).strip().upper()   # 选择题
        category = ex.get("category") if "category" in ex else None

        if options: # options有的话，就跟question链接。
            prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer:"
        else:       # 无的话选项就在question里
            prompt = f"{question}\n\nAnswer:"

        sample["prompt"]   = prompt
        sample["label"]    = label
        sample["category"] = category   # Only BBH has category

        dataset.append(sample)
    
    return dataset

if __name__ == "__main__":
    Dir = "/mnt/zhangchen/S3Precision/LLM-inference/Datasets/"
    dataset = process_data(Dir + "BBH/test.json")
    print(dataset[0], len(dataset))
    dataset = process_data(Dir + "GPQA-diamond/test.json")
    print(dataset[0], len(dataset))
    dataset = process_data(Dir + "MMLU-PRO/test.json")
    print(dataset[0], len(dataset))
