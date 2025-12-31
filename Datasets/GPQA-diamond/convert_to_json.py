import pandas as pd
import json
import numpy as np
import string

df = pd.read_parquet("/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/datasets/GPQA-diamond/gpqa_diamond.parquet")

json_data = []
for _, row in df.iterrows():
    question = row["question"]
    answer = row["answer"]

    entry = {
        "instruction": (
            "You are solving a multiple-choice question.Select the correct answer and answerw with only the correct option's letter\n\nQuestion:\n" + question.strip()
        ),
        "input": "",
        "output": str(answer).strip()
    }
    json_data.append(entry)

with open("/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/datasets/GPQA-diamond/test.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(json_data)}")