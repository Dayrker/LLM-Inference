import json
import csv

# Input and output paths
json_path = "/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/datasets/BBH/data.json"
csv_path = "/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/datasets/BBH/data.csv"

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Write CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(["instruction", "input", "output"])
    # Write rows
    for entry in data:
        writer.writerow([entry.get("instruction", ""), entry.get("input", ""), entry.get("output", "")])

print(f"✅ JSON converted to CSV successfully: {csv_path}")
