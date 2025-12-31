import re

def benchmark_MMLU(answer: str, label: str):
    if not answer:
        return "UNKNOWN", False

    text = answer.strip()

    explicit_pattern = re.compile(
        r"(?:The\s+)?[Cc]orrect\s+[Aa]nswer\s*(?:is)?\s*[:\-]?\s*([A-J])\b|"
        r"[Aa]nswer\s*[:\-]?\s*([A-J])\b"
    )

    match = explicit_pattern.search(text)
    if match:
        prediction = (match.group(1) or match.group(2)).upper().strip()
    else:
        fallback_pattern = re.compile(
            r'(?:^|[\s,:;()\-])(?:[Oo]ption\s*)?([A-J])(?:[\s,:;()\-]|$)'
        )
        fallback_match = fallback_pattern.search(text)
        if fallback_match:
            prediction = fallback_match.group(1).upper().strip()
        else:
            prediction = "UNKNOWN"

    correct = prediction == label
    return prediction, correct