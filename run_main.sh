python main.py \
--cuda "0, 1, 2, 3, 4, 5, 6, 7" \
--model "Qwen/Qwen3-8B" \
--arch "NV" \
--precision "nvfp4" \
--dataset "BBH" \
--batch_size 32