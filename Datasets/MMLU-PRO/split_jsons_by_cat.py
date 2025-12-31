import os
import json
import string
import pandas as pd

# === CONFIG ===
input_parquet = "/mnt/alvaro/projects/llm-training-inference-main/wrap-to-fp8/datasets/MMLU-PRO/test-00000-of-00001.parquet"           # path to your MMLU-Pro parquet
output_folder = "/mnt/alvaro/projects/llm-training-inference-main/wrap-to-fp8/datasets/MMLU-PRO/categories"
os.makedirs(output_folder, exist_ok=True)

# === LOAD DATA ===
df = pd.read_parquet(input_parquet)
print(f"Loaded {len(df)} rows from {input_parquet}")

# Ensure required columns exist
required_cols = {"question", "options", "answer", "category"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# === GROUP BY CATEGORY ===
for category, group in df.groupby("category"):
    json_data = []

    for _, row in group.iterrows():
        question = str(row["question"]).strip()
        options = row["options"]
        answer = str(row["answer"]).strip()

        # ensure options are a list
        if not isinstance(options, list):
            try:
                options = list(options)
            except Exception:
                options = [str(options)]

        # label options as A), B), C), ...
        labeled_options = "\n".join(
            f"{letter}) {opt}" for letter, opt in zip(string.ascii_uppercase, options)
        )

        # build the same prompt structure
        entry = {
            "instruction": question + "\n\nChoose the correct option below:",
            "input": labeled_options,
            "output": answer,
        }
        json_data.append(entry)

    # sanitize filename
    safe_name = category.lower().replace(" ", "_").replace("/", "_")
    out_path = os.path.join(output_folder, f"{safe_name}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(json_data)} examples to {out_path}")

print(f"\n📁 All category files saved under: {output_folder}")
