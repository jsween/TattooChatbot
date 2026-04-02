import re

def normalize_text(text):
    text = text.lower().strip()
    # common
    replacements = {
        r"\bu\b": "you",
        r"\bur\b": "your",
        r"\bappt\b": "appointment",
        r"\bavail\b": "availability",
        r"\bloc\b": "location",
        r"\baddr\b": "address",
        r"\binfo\b": "information",
        r"\bwa\b": "washington",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # normalize repeated punctuation
    text = re.sub(r"[!?]+", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text