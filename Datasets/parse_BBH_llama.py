import re

def benchmark_BBH_llama(answer: str, label: str, category: str = None):
    """
    Benchmark function for evaluating BBH-style tasks across multiple categories.
    
    Dynamically parses model outputs depending on task type (multiple choice, numerical, etc.),
    normalizes answers, and compares them to labels.
    
    Args:
        answer (str): Model-generated output text.
        label (str): Ground-truth label for comparison.
        category (str, optional): Type of question, determines parsing logic.
            Supported categories:
                - "multiple_choice" (A–J)
                - "true_false"
                - "yes_no"
                - "numerical"
                - "valid_invalid"
                - "words_sorted"
                - "sequence"
                - None → auto-detects
        
    Returns:
        dict: {
            "predicted": str,
            "correct": bool,
            "category": str,
            "confidence": float (heuristic confidence 0–1),
            "raw_text": str
        }
    """
    
    if not answer or not isinstance(answer, str):
        return {"predicted": "UNKNOWN", "correct": False, "category": category or "unknown", "confidence": 0.0, "raw_text": answer}

    cleaned = answer.strip()
    category = category or _auto_detect_category(cleaned)
    
    # Dispatch to appropriate parser
    parsers = {
        "multiple_choice": _parse_multiple_choice,
        "true_false": _parse_true_false,
        "yes_no": _parse_yes_no,
        "numerical": _parse_numerical,
        "valid_invalid": _parse_valid_invalid,
        "words_sorted": _parse_words_sorted,
        "sequence": _parse_sequence
    }
    
    parser = parsers.get(category, _parse_multiple_choice)
    predicted = parser(cleaned)
    predicted = predicted.strip() if predicted else "UNKNOWN"
    
    correct = _compare(predicted, label, category)
    confidence = _estimate_confidence(cleaned, predicted, category)
    
    return {
        "predicted": predicted,
        "correct": correct,
        "category": category,
        "confidence": confidence,
        "raw_text": cleaned
    }


# --------------------------
# Parsing utilities
# --------------------------

def _auto_detect_category(text: str) -> str:
    """Infer category based on content heuristics."""
    if re.search(r'\b(True|False)\b', text, re.IGNORECASE):
        return "true_false"
    if re.search(r'\b(Yes|No)\b', text, re.IGNORECASE):
        return "yes_no"
    if re.search(r'\b(valid|invalid)\b', text, re.IGNORECASE):
        return "valid_invalid"
    if re.search(r'\b[A-J]\b', text):
        return "multiple_choice"
    if re.search(r'\b\d+(?:\.\d+)?\b', text):
        return "numerical"
    if re.search(r'[{}()<>\[\]]', text):
        return "sequence"
    return "words_sorted"


def _parse_multiple_choice(text: str) -> str:
    """Extract multiple-choice answers (A–J)."""
    patterns = [
        r'(?:[Tt]he\s+)?[Cc]orrect\s+[Aa]nswer\s*(?:is)?\s*[:\-]?\s*([A-J])\b',
        r'[Aa]nswer\s*[:\-]?\s*([A-J])\b',
        r'[Oo]ption\s*[:\-]?\s*([A-J])\b',
        r'\b([A-J])[\)\.]', 
        r'[\(\[]([A-J])[\)\]]',
        r'\b([A-J])\b(?!\w)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return ""


def _parse_true_false(text: str) -> str:
    match = re.search(r'\b(True|False)\b', text, re.IGNORECASE)
    return match.group(1).capitalize() if match else ""


def _parse_yes_no(text: str) -> str:
    match = re.search(r'\b(Yes|No)\b', text, re.IGNORECASE)
    return match.group(1).capitalize() if match else ""


def _parse_valid_invalid(text: str) -> str:
    match = re.search(r'\b(valid|invalid)\b', text, re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _parse_numerical(text: str) -> str:
    match = re.search(r'[-+]?\d*\.?\d+', text)
    return match.group(0) if match else ""


def _parse_words_sorted(text: str) -> str:
    """Return a line where words appear alphabetically sorted or most structured."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in reversed(lines):
        words = line.split()
        if len(words) > 1:
            sorted_words = sorted(words)
            if words == sorted_words:
                return line
            if not any(prefix in line.lower() for prefix in ['answer:', 'final answer:', 'result:']):
                return line
    return lines[-1] if lines else ""


def _parse_sequence(text: str) -> str:
    match = re.search(r'Answer[:\s]*([^\.\n]+)', text)
    if match:
        return match.group(1).strip()
    brackets = [line for line in text.splitlines() if any(c in line for c in '{}[]()<>')]
    return brackets[-1].strip() if brackets else ""


# --------------------------
# Comparison & Confidence
# --------------------------

def _compare(predicted: str, label: str, category: str) -> bool:
    """Normalize and compare based on category."""
    if not predicted or not label:
        return False
    
    predicted_norm = predicted.strip().lower()
    label_norm = label.strip().lower()
    
    if category in {"true_false", "yes_no", "valid_invalid"}:
        return predicted_norm == label_norm
    if category == "multiple_choice":
        return predicted_norm == label_norm
    if category == "numerical":
        try:
            return abs(float(predicted_norm) - float(label_norm)) < 1e-6
        except ValueError:
            return predicted_norm == label_norm
    return predicted_norm == label_norm


def _estimate_confidence(text: str, predicted: str, category: str) -> float:
    """Heuristic confidence estimation based on match strength and explicitness."""
    if predicted == "UNKNOWN" or not predicted:
        return 0.1
    if re.search(r'[Cc]orrect\s+[Aa]nswer', text):
        return 0.95
    if category == "multiple_choice" and re.search(r'\b[A-J]\b', text):
        return 0.8
    if category in {"true_false", "yes_no"} and re.search(predicted, text, re.IGNORECASE):
        return 0.85
    if category == "numerical" and re.search(r'[-+]?\d*\.?\d+', text):
        return 0.8
    return 0.6
