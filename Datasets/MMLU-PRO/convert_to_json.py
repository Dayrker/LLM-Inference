import pandas as pd
import json
import numpy as np
import string

df = pd.read_parquet("/mnt/alvaro/projects/llm-training-inference-main/wrap-to-fp8/datasets/MMLU-PRO/test-00000-of-00001.parquet")

json_data = []
for _, row in df.iterrows():
    question = row["question"]
    options = row["options"]
    answer = row["answer"]

    if not isinstance(options, list):
        options = list(options)

    labeled_options = "\n".join(
        f"{letter}) {opt}" for letter, opt in zip(string.ascii_uppercase, options)
    )

    entry = {
        "instruction": (
            "You are solving a multiple-choice question.Select the correct answer and write only its letter\n\nQuestion:\n" + question.strip()
        ),
        "input": "Options:\n" + labeled_options,
        "output": str(answer).strip()
    }
    json_data.append(entry)

with open("/mnt/alvaro/projects/llm-training-inference-main/wrap-to-fp8/datasets/MMLU-PRO/test_2.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(json_data)}")