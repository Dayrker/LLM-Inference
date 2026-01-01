python main.py \
--cuda "0, 1, 2, 3, 4, 5, 6, 7" \
--model "llama/Llama-3-8B-Instruct" \
--arch "NV" \
--precision "nvfp4" \
--dataset "MMLU-PRO" \
--batch_size 8