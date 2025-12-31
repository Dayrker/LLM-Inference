import os
import json
from pathlib import Path
import re

bbh_folder = Path("/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/datasets/BBH/categories")
output_file = Path("/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/datasets/BBH/data_classified.json")

output_file.parent.mkdir(parents=True, exist_ok=True)

all_data = []
split_keywords = ["Options:", "Input:", "List:"]

suffix_rules = {
    "boolean_expressions": {
        "suffix": "You have to answer the following question. Answer only with True or False. Do not add explanation.\n\nQuestion:\n",
        "category": "true_false"
    },
    "causal_judgement": {
        "suffix": "You have to answer the following question. Answer only with Yes or No. Do not add explanation.\n\nQuestion:\n",
        "category": "yes_no"
    },
    "sports_understanding": {
        "suffix": "You have to answer the following question. Answer only with Yes or No. Do not add explanation.\n\nQuestion:\n",
        "category": "yes_no"
    },
    "web_of_lies": {
        "suffix": "You have to answer the following question. Answer only with Yes or No. Do not add explanation.\n\nQuestion:\n",
        "category": "yes_no"
    },
    "navigate": {
        "suffix": "You have to answer the following question. Answer only with Yes or No. Do not add explanation.\n\nQuestion:\n",
        "category": "yes_no"
    },
    "dyck_languages": {
        "suffix": "You have to answer the following question. Answer only with the missing parentheses. Do not add explanation.\n\nQuestion:\n",
        "category": "sequence"
    },
    "formal_fallacies": {
        "suffix": "You have to answer the following question. Answer only with 'valid' or 'invalid'. Do not add explanation.\n\nQuestion:\n",
        "category": "valid_invalid"
    },
    "multistep_arithmetic_two": {
        "suffix": "You have to answer the following question. Answer only with the final result. Do not add explanation.\n\nQuestion:\n",
        "category": "numerical"
    },
    "object_counting": {
        "suffix": "You have to answer the following question. Answer only with the final result. Do not add explanation.\n\nQuestion:\n",
        "category": "numerical"
    },
    "word_sorting": {
        "suffix": "You have to answer the following question. Answer only with the words sorted. Do not add explanation.\n\nQuestion:\n",
        "category": "words_sorted"
    }
}

default_config = {
    "suffix": "You have to answer the following question. Answer only with the letter of the correct option. Do not add explanation.\n\nQuestion:\n",
    "category": "multiple_choice"
}

for json_file in sorted(bbh_folder.glob("*.json")):
    dataset_name = json_file.stem

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    for ex in examples:
        raw_input = ex.get("input", "").strip()
        target = str(ex.get("target", "")).strip()

        instruction, input_part = raw_input, ""
        is_options_case = False

        for kw in split_keywords:
            if kw in raw_input:
                parts = raw_input.split(kw, 1)
                instruction = parts[0].strip()
                input_part = parts[1].strip()
                if kw == "Options:":
                    is_options_case = True
                break

        if is_options_case and target.startswith("(") and target.endswith(")"):
            target = target[1:-1].strip()

        config = suffix_rules.get(dataset_name, default_config)
        suffix = config["suffix"]
        category = config["category"]
        
        if not instruction.endswith(suffix):
            instruction = f"{suffix}{instruction.strip()}"

        entry = {
            "instruction": instruction,
            "input": input_part,
            "output": target,
            "category": category
        }
        all_data.append(entry)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"Saved formatted BBH dataset to: {output_file}")