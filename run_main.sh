python main.py \
--cuda "0, 1, 2, 3, 4, 5, 6, 7" \
--model "Qwen/Qwen3-8B" \
--arch "NV" \
--precision "baseline" \
--dataset "BBH" \
--batch_size 8