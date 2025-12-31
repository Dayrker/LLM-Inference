import os
import json
from pathlib import Path
import re

# --- Input/output paths ---
bbh_folder = Path("/mnt/alvaro/projects/llm-training-inference-main/wrap-to-fp8/datasets/BBH/categories")
output_file = Path("/mnt/alvaro/projects/llm-training-inference-main/wrap-to-fp8/datasets/BBH/data.json")

output_file.parent.mkdir(parents=True, exist_ok=True)

all_data = []
split_keywords = ["Options:", "Input:", "List:"]

# --- Instruction suffixes by dataset name ---
suffix_rules = {
    "boolean_expressions": "You have to answer the following question. Answer only with True or False. Do not add explanation.\n\nQuestion:\n",
    "causal_judgement": "You have to answer the following question. Answer only with Yes or No. Do not add explanation.\n\nQuestion:\n",
    "sports_understanding": "You have to answer the following question. Answer only with Yes or No. Do not add explanation.\n\nQuestion:\n",
    "web_of_lies": "You have to answer the following question. Answer only with Yes or No. Do not add explanation.\n\nQuestion:\n",
    "navigate": "You have to answer the following question. Answer only with Yes or No. Do not add explanation.\n\nQuestion:\n",
    "dyck_languages": "You have to answer the following question. Answer only with the missing parentheses. Do not add explanation.\n\nQuestion:\n",
    "formal_fallacies": "You have to answer the following question. Answer only with 'valid' or 'invalid'. Do not add explanation.\n\nQuestion:\n",
    "multistep_arithmetic_two": "You have to answer the following question. Answer only with the final result. Do not add explanation.\n\nQuestion:\n",
    "object_counting": "You have to answer the following question. Answer only with the final result. Do not add explanation.\n\nQuestion:\n",
    "word_sorting": "You have to answer the following question. Answer only with the words sorted. Do not add explanation.\n\nQuestion:\n"
}

default_suffix = "You have to answer the following question. Answer only with the letter of the correct option. Do not add explanation.\n\nQuestion:\n"

# --- Process all BBH JSON files ---
for json_file in sorted(bbh_folder.glob("*.json")):
    dataset_name = json_file.stem  # filename without extension

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    for ex in examples:
        raw_input = ex.get("input", "").strip()
        target = str(ex.get("target", "")).strip()

        instruction, input_part = raw_input, ""
        is_options_case = False

        # --- Split by keyword (first match only) ---
        for kw in split_keywords:
            if kw in raw_input:
                parts = raw_input.split(kw, 1)
                instruction = parts[0].strip()
                input_part = parts[1].strip()
                if kw == "Options:":
                    is_options_case = True
                break

        # --- Clean target if options-type (e.g., "(A)" → "A") ---
        if is_options_case and target.startswith("(") and target.endswith(")"):
            target = target[1:-1].strip()

        # --- Append suffix based on file name ---
        suffix = suffix_rules.get(dataset_name, default_suffix)
        if not instruction.endswith(suffix):
            instruction = f"{suffix}{instruction.strip()}"

        entry = {
            "instruction": instruction,
            "input": input_part,
            "output": target
        }
        all_data.append(entry)

print(f"✅ Processed {len(all_data)} total examples from {len(list(bbh_folder.glob('*.json')))} files.")

# --- Save unified dataset ---
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"💾 Saved formatted BBH dataset to: {output_file}")
