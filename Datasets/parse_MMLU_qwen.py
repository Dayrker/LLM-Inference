import re

def benchmark_MMLU_qwen(answer: str, label: str):
    if not answer:
        return "UNKNOWN", False

    text = answer.strip()

    option_pattern = r"\b([A-J])(?:\)|\.|:|\b)"

    think_split = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(think_split) > 1:
        after_think = think_split[1]
        match = re.search(option_pattern, after_think)
        if match:
            prediction = match.group(1).upper()
            return prediction, prediction == label

    answer_split = re.split(r"[Aa]nswer", text, maxsplit=1)
    if len(answer_split) > 1:
        after_answer = answer_split[1]
        match = re.search(option_pattern, after_answer)
        if match:
            prediction = match.group(1).upper()
            return prediction, prediction == label

    fallback_match = re.search(option_pattern, text)
    if fallback_match:
        prediction = fallback_match.group(1).upper()
    else:
        prediction = "UNKNOWN"

    correct = prediction == label
    return prediction, correct
