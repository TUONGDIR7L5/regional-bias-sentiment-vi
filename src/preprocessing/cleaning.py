import html
import re
import unicodedata

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U0001F900-\U0001F9FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

ABBREVIATION_MAP = {
    "ko": "không", "k": "không", "kg": "không", "kh": "không", "hok": "không",
    "dc": "được", "đc": "được", "j": "gì", "z": "vậy", "vs": "với",
    "ng": "người", "nhg": "nhưng", "nma": "nhưng mà", "bt": "bình thường",
    "bik": "biết", "bít": "biết", "ny": "người yêu", "cx": "cũng",
    "sn": "sinh nhật", "oki": "ok",
    "mn": "mọi người", "mng": "mọi người", "ae": "anh em", "cmt": "bình luận",
    "add": "add", "sv": "sinh viên", "hs": "học sinh",
    "tk": "tài khoản", "nc": "nói chuyện", "thanks": "cảm ơn", "thank": "cảm ơn",
    "tks": "cảm ơn", "kb": "kết bạn", "onl": "online", "off": "offline",
}


def remove_emoji(text):
    return EMOJI_PATTERN.sub(" ", text)


def expand_abbreviations(text):
    tokens = text.split(" ")
    expanded = []
    for tok in tokens:
        key = tok.lower().strip(".,!?;:")
        if key in ABBREVIATION_MAP:
            expanded.append(ABBREVIATION_MAP[key])
        else:
            expanded.append(tok)
    return " ".join(expanded)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = remove_emoji(text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_sentence(text, min_words=3, max_words=80):
    if not text:
        return False
    n = len(text.split())
    return min_words <= n <= max_words
