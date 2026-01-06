# need specific env for DeepSeek-OCR
# -> https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/requirements.txt
python main_ocr.py \
--cuda "0" \
--model "DeepSeek/DeepSeek-OCR" \
--arch "NV" \
--precision "mxfp8" \
--dataset "OmniDocBench" \
--batch_size 1