import re

def normalize_text(text):
    text = text.lower().strip()
    # common slang or shortened words
    replacements = {
        r"\bu\b": "you",
        r"\bur\b": "your",
        r"\bappt\b": "appointment",
        r"\bavail\b": "availability",
        r"\bloc\b": "location",
        r"\baddr\b": "address",
        r"\baddy\b": "address",
        r"\binfo\b": "information",
        r"\bwa\b": "washington",
        r"\bwanna\b": "want to",
        r"\bgonna\b": "going to",
        r"\bidk\b": "i do not know",
        r"\bpls\b": "please",
        r"\bthx\b": "thanks",
        r"\bprice\??\b": "pricing",
        r"\bcost\??\b": "pricing",
        r"\bhow much\b": "pricing",
        r"\bhow long\b": "time estimate",
        r"\bbook\b": "booking",
        r"\bschedule\b": "booking",
        r"\bappointment\b": "booking",
        r"\bwhere are you\b": "location",
        r"\bsecond skin\b": 'aftercare',
        r"\b(\d+)\s*inch(es)?\b": r"\1 inch",
        r"\bsmall\b": "small size",
        r"\bmedium\b": "medium size",
        r"\blarge\b": "large size",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # normalize repeated punctuation
    text = re.sub(r"[!?]+", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text