import re
import unicodedata


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # bỏ URL
    text = re.sub(r"@\w+", " ", text)               # bỏ mention
    text = re.sub(r"\s+", " ", text).strip()        # gộp khoảng trắng
    return text
