import os
import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        required=True,
        help="device list (like \"0, 1, 2, 3\").",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama/Llama-3-8B-Instruct",
        required=True,
        help="model name (like llama/Llama-3-8B-Instruct).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="NV",
        required=True,
        help="architecture (NV/DW).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="baseline",
        required=True,
        help="precision for inference (baseline/mxfp8/nvfp4).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MMLU-PRO",
        required=True,
        help="datasets selection (BBH/GPQA-diamond/MMLU-PRO).",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  # Set cuda devices first.
    return args