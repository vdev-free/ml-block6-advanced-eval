import re

def normalize_text_basic(text: str) -> str:
    text = text.lower()
    text = text.strip()
    return text

def remove_punctuation(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    return text

def remove_numbers(text: str) -> str:
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

def preprocess_text(text: str, remove_digits: bool = False) -> str:
    text = normalize_text_basic(text)
    text = remove_punctuation(text)

    if remove_digits:
        text = remove_numbers(text)

    text = " ".join(text.split())
    return text

