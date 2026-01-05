# utils
from utils.helpers import save_data_to_json
from utils.parse import parse_args
# Datasets
from Datasets.load_data import process_data
from Datasets.compute_metric import compute_metrics
# Inference
from Inference.infer_batch import infer_batch, infer_batch_multiprocessing

if __name__ == "__main__":
    # Get parameters first.
    args = parse_args()

    # # get model
    # model, tokenizer = getModel("/ssd/models/" + args.model)

    # get datasets
    data_dir = "/mnt/zhangchen/S3Precision/LLM-inference/Datasets/"
    datasets = process_data(data_dir + args.dataset + "/test.json")

    # get outputs
    if len(args.cuda.split(",")) <= 1:
        outputs = infer_batch(args, datasets, device=f"cuda:{int(args.cuda)}")
    else:
        outputs = infer_batch_multiprocessing(args, datasets)

    # compute metrics & save
    metrics = compute_metrics(outputs, datasets, args.dataset, args.model)
    save_data_to_json(metrics, f"./results/{args.model}/{args.arch}/{args.precision}/result_{args.dataset}.json")
