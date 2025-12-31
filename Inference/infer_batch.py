from tqdm import tqdm
# Parallel
import torch.multiprocessing as mp
# convert model
from .convert_model import replace_modules

# ---------------- Helper for chat template ----------------
def toChat(tokenizer, p: str) -> str:
    messages = [{"role": "user", "content": p}]

    chat_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return chat_text

def infer_batch(llm_model, tokenizer, dataset, args,
                device = "cuda:0", return_queue = None):
    ### parse parameters needed
    arch       = args.arch
    precision  = args.precision
    batch_size = args.batch_size
    # model preprocess
    llm_model.to(device).eval()
    replace_modules(llm_model, arch = arch, precision = precision)   # replace_modules必须挪到这里来，因为TE内部有局部函数/闭包，无法被spawn序列化

    dataLen = len(dataset)
    outputs = []
    # ---------------- Batch inference ----------------
    for batch_start in tqdm(range(0, dataLen, batch_size), desc="Running inference", ncols=100):
        # text pre-process
        batch_data = dataset[batch_start: batch_start + batch_size]
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
            max_new_tokens=128,    # 生成长度，后续可指定 (alvaro -> 70 for MMLU and BBH, 130 for the rest)
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

    if return_queue:    # 返回本GPU结果，用于multi-gpu
        return_queue.put(outputs)
    return outputs


def infer_batch_multiprocessing(llm_model, tokenizer, dataset, args):
    ### parse args needed
    device     = args.cuda

    dataset_len = len(dataset)
    world_size = len(device.split(","))

    # split dataset
    data_slices = []
    per_gpu = (dataset_len + world_size - 1) // world_size
    for i in range(world_size):
        start = i * per_gpu
        end   = min(start + per_gpu, dataset_len)
        data_slices.append(dataset[start: end])

    # === 启动多进程 ===
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_queue = manager.Queue()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target = infer_batch,
            args = (llm_model, tokenizer, data_slices[rank], args,
                    f"cuda:{rank}", return_queue)
        )
        p.start()
        processes.append(p)
    for p in processes: p.join()    # 阻塞主进程，等待这个子进程结束 -> 等待所有 GPU 完成

    # === 收集4卡结果 ===
    all_results = []
    while not return_queue.empty():
        all_results.extend(return_queue.get())
    # === 按 index 排序 ===
    all_results = sorted(all_results, key=lambda x: x["id"])

    print(f"🔥 已完成 {world_size}-GPU 推理.")
    # print("all_results:", all_results)
    return all_results

if __name__ == "__main__":
    outputs = infer_batch()