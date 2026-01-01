# utils
from utils.helpers import same_seed, getContent, save_data_to_json
from utils.parse import parse_args
# Models
from Models.LoadModel import getModel
# Datasets
from Datasets.load_data import process_data
from Datasets.compute_metric import compute_metrics
# Inference
from Inference.infer_batch import infer_batch, infer_batch_multiprocessing

if __name__ == "__main__":
    # Get parameters first.
    args = parse_args()
    same_seed(42)

    # get model
    model, tokenizer = getModel("/ssd/models/" + args.model)

    # get datasets
    data_dir = "/mnt/zhangchen/S3Precision/LLM-inference/Datasets/"
    datasets = process_data(data_dir + args.dataset + "/test.json")

    content = getContent(arch=args.arch, precision=args.precision)
    with content:
        # get outputs
        outputs = infer_batch_multiprocessing(model, tokenizer, datasets, args)

    # compute metrics & save
    metrics = compute_metrics(outputs, datasets, args.dataset)
    save_data_to_json(metrics, f"./results/{args.arch}/{args.precision}/result_{args.dataset}.json")
