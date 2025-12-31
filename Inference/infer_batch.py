from tqdm import tqdm

# ---------------- Helper for chat template ----------------
def toChat(tokenizer, p: str) -> str:
    messages = [{"role": "user", "content": p}]

    chat_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return chat_text

def infer_batch(llm_model, tokenizer, 
                dataset, batch_size = 8,
                device="cuda:0"):
    llm_model.eval()
    dataLen = len(dataset)

    # ---------------- Batch inference ----------------
    for batch_start in tqdm(range(0, dataLen, batch_size), desc="Running inference", ncols=100):
        # text pre-process
        texts = [toChat(tokenizer, x["prompt"]) for x in dataset[batch_start:batch_start + batch_size]]
        # tokenize
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,    # 截断长度，后续可指定
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        print(inputs)

    return outputs

if __name__ == "__main__":
    outputs = infer_batch()