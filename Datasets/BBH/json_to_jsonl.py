import json

# Input and output paths
input_json_path = "/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/datasets/BBH/data.json"
output_jsonl_path = "/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/datasets/BBH/data.jsonl"

# Read the JSON file (expects a list of objects)
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Write as JSONL
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Converted {input_json_path} to {output_jsonl_path} with {len(data)} lines.")
