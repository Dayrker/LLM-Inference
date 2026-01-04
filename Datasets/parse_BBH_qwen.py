import re

def benchmark_BBH_qwen(answer: str, label: str, category: str = None):
    """
    Enhanced BBH benchmark parser that extracts the model's final answer
    **only after the literal '</think>' tag** (case-insensitive),
    then auto-detects question type and evaluates correctness.
    """

    if not answer or not isinstance(answer, str):
        return {
            "predicted": "UNKNOWN",
            "correct": False,
            "category": category or "unknown",
            "raw_text": answer,
        }

    # --- Step 1: Normalize and clean ---
    text = answer.replace("/nothink", "").replace("\\n", "\n").strip()

    # --- Step 2: Extract only what comes after </think> ---
    if "</think>" in text.lower():
        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        after_think = parts[1] if len(parts) > 1 else text
    else:
        after_think = text  # fallback if model didn't output </think>

    # --- Step 3: Clean trailing junk and empty lines ---
    after_think = re.sub(r"<[^>]+>", "", after_think)  # remove tags like <think>
    after_think = after_think.strip()
    lines = [line.strip() for line in after_think.splitlines() if line.strip()]
    final_text = "\n".join(lines) if lines else after_think

    # --- Step 4: Use only the last or first meaningful line ---
    # Most model outputs end with "True", "B", "42", etc.
    if lines:
        candidate = lines[-1]
    else:
        candidate = final_text

    # --- Step 5: Auto-detect category if not given ---
    cat = (category or _auto_detect_category(candidate)).lower()

    # --- Step 6: Parse according to category ---
    predicted = _parse_by_category(candidate, cat)

    # --- Step 7: Normalize + Compare ---
    correct = _compare(predicted, label, cat)
    return {
        "predicted": predicted or "UNKNOWN",
        "correct": correct,
        "category": cat,
        "raw_text": after_think,
    }


# --------------------------
# Category detection & parsing
# --------------------------

def _auto_detect_category(text: str) -> str:
    text_lower = text.lower()
    if re.search(r"\b(true|false)\b", text_lower):
        return "true_false"
    if re.search(r"\b(yes|no)\b", text_lower):
        return "yes_no"
    if re.search(r"\b(valid|invalid)\b", text_lower):
        return "valid_invalid"
    if re.search(r"\b[A-J]\b", text_lower):
        return "multiple_choice"
    if re.search(r"\b\d+(?:\.\d+)?\b", text_lower):
        return "numerical"
    if re.search(r"[{}()<>\[\]]", text_lower):
        return "sequence"
    return "text"


def _parse_by_category(text: str, cat: str) -> str:
    text = text.strip()
    prediction = "UNKNOWN"

    if cat in ["true_false", "boolean"]:
        m = re.search(r"\b(true|false)\b", text, re.IGNORECASE)
        if m:
            prediction = m.group(1).capitalize()

    elif cat == "yes_no":
        m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
        if m:
            prediction = m.group(1).capitalize()

    elif cat == "valid_invalid":
        m = re.search(r"\b(valid|invalid)\b", text, re.IGNORECASE)
        if m:
            prediction = m.group(1).lower()

    elif cat == "multiple_choice":
        m = re.search(r"\b([A-J])\b", text)
        if m:
            prediction = m.group(1).upper()

    elif cat == "numerical":
        m = re.search(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
        if m:
            prediction = m.group(0).strip()

    elif cat == "sequence":
        m = re.search(r"Answer[:\s]*([^\.\n]+)", text)
        if m:
            prediction = m.group(1).strip()
        else:
            brackets = [line for line in text.splitlines() if any(c in line for c in "{}[]()<>")]
            if brackets:
                prediction = brackets[-1].strip()

    else:
        prediction = text.strip()

    return prediction


def _compare(predicted: str, label: str, category: str) -> bool:
    if not predicted or not label:
        return False

    p, l = predicted.strip().lower(), label.strip().lower()

    if category in {"true_false", "yes_no", "valid_invalid", "multiple_choice"}:
        return p == l

    if category == "numerical":
        try:
            return abs(float(p) - float(l)) < 1e-6
        except ValueError:
            return p == l

    return p == l
