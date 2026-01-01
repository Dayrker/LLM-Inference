from tqdm import tqdm
from .parse_BBH import benchmark_BBH
from .parse_MMLU import benchmark_MMLU
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(outputs, dataset, dataset_name):
    assert dataset_name in ["BBH", "MMLU-PRO", "GPQA-diamond"], f"Metrics for dataset \"{dataset_name}\" not implemented."

    dataLen = len(dataset)
    for i in tqdm(range(dataLen), desc="Computing metrics", ncols=100):
        answer   = outputs[i]["answer"]
        label    = dataset[i]["label"]
        category = dataset[i]["category"]

        if dataset_name == "BBH":
            parsed = benchmark_BBH(answer, label, category)
            prediction, correct = parsed["predicted"], parsed["correct"]
        else:   # MMLU-PRO and GPQA-diamond
            prediction, correct = benchmark_MMLU(answer, label)
        
        # Rewrite metrics to dataset
        dataset[i]["answer"]     = outputs[i]["answer"]
        dataset[i]["prediction"] = prediction
        dataset[i]["correct"]    = correct
    
    # Compute overall metrics
    labels      = [data["label"]      for data in dataset]
    predictions = [data["prediction"] for data in dataset]
    acc = accuracy_score(labels, predictions)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    dataset.append({
        "accuracy": acc,
        "precision_macro": prec_macro,
    })
    print(f"📌 汇总指标: accuracy: {round(acc, 4)}, precision_macro: {round(prec_macro, 4)}")
    return dataset
