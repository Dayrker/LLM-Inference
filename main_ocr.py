# utils
from utils.helpers import save_data_to_json
from utils.parse import parse_args
# Datasets
from Datasets.load_data import process_ocr_data
# Inference
from Inference.infer_batch_ocr import infer_batch_ocr   # , infer_batch_ocr_multiprocessing


if __name__ == "__main__":
    # Get parameters first.
    args = parse_args()
    
    # get datasets
    datasets = process_ocr_data()
    
    # get outputs
    if len(args.cuda.split(",")) <= 1:
        outputs = infer_batch_ocr(args, datasets, device=f"cuda:{int(args.cuda)}")
    # else:
    #     outputs = infer_batch_ocr_multiprocessing(args, datasets)