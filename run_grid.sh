# python main.py \
# --cuda "0, 1, 2, 3, 4, 5, 6, 7" \
# --model "llama/Llama-3-8B-Instruct" \
# --arch "DW" \
# --precision "baseline" \
# --dataset "MMLU-PRO" \
# --batch_size 8


CUDA="0,1,2,3,4,5,6,7"
BATCH_SIZE=8

MODELS=(
  # "llama/Llama-3-8B-Instruct"
  "Qwen/Qwen3-8B"
)
ARCHS=(
  "NV"
  "DW"
)
PRECISIONS=(
  "baseline"
  "mxfp8"
  "nvfp4"
)
DATASETS=(
  "MMLU-PRO"
  "BBH"
  "GPQA-diamond"
)

for MODEL in "${MODELS[@]}"; do
  for ARCH in "${ARCHS[@]}"; do
    for PRECISION in "${PRECISIONS[@]}"; do
      for DATASET in "${DATASETS[@]}"; do
        echo "Running: model=${MODEL}, arch=${ARCH}, precision=${PRECISION}, dataset=${DATASET}"
        python main.py \
          --cuda "${CUDA}" \
          --model "${MODEL}" \
          --arch "${ARCH}" \
          --precision "${PRECISION}" \
          --dataset "${DATASET}" \
          --batch_size ${BATCH_SIZE}
      done
    done
  done
done
