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

    outputs = []
    # ---------------- Batch inference ----------------
    for batch_start in tqdm(range(0, dataLen, batch_size), desc="Running inference", ncols=100):
        # text pre-process
        batch_data = dataset[batch_start: batch_start + batch_size]
        print("batch_data:", batch_data)
        texts      = [toChat(tokenizer, x["prompt"]) for x in batch_data]
        # tokenize
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,    # 截断长度，后续可指定
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)   # [0/1, ...] -> 0/1 list (0代表mask, 1代表正常token)

        # generate
        out_ids = llm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=1024,    # 生成长度，后续可指定
        )
        promptLen = input_ids.shape[1]  # tokenizer有设置padding_side='left'，所以生成的prompt都会在input_ids最右边生成，起始点一致
        answers = tokenizer.batch_decode(out_ids[:, promptLen:], skip_special_tokens=True)

        # preces output
        for i in range(len(texts)):
            output = {
                "id": batch_data[i]["id"]
            }
            output["answer"] = answers[i]
            outputs.append(output)

    return outputs

if __name__ == "__main__":
    outputs = infer_batch()