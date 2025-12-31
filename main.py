# utils
from utils.helpers import same_seed, replace_modules, getContent
from utils.parse import parse_args
# Models
from Models.LoadModel import getModel
# Datasets
from Datasets.LoadData import process_data
from Datasets.parse_BBH import benchmark_BBH
from Datasets.parse_MMLU import benchmark_mmlu
# Inference
from Inference.infer_batch import infer_batch, infer_batch_multiprocessing

if __name__ == "__main__":
    # Get parameters first.
    args = parse_args()
    same_seed(42)

    # get model
    model, tokenizer = getModel("/ssd/models/" + args.model)
    replace_modules(model, arch=args.arch, precision=args.precision)
    print("model:", model)

    # get datasets
    data_dir = "/mnt/zhangchen/S3Precision/LLM-inference/Datasets/"
    datasets = process_data(data_dir + args.dataset + "/test.json")

    content = getContent(arch=args.arch, precision=args.precision)
    with content:
        # get outputs
        outputs = infer_batch_multiprocessing(model, tokenizer, datasets, args.batch_size, args.cuda)
    
    # save results