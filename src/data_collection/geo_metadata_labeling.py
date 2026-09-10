import re
import unicodedata

REGION_LEXICON = {
    "northern": [
        "giời", "ối giời ơi", "u ơi", "bu ơi", "thầy u", "chả biết", "chả có", "chả là",
        "đấy nhở", "cơ mà", "thế cơ đấy", "cơ đấy", "giả nhời", "vưỡn", "nhở",
        "mà lị", "chứ lị", "đằng ấy", "hay phết", "giỏi phết", "đẹp phết", "ra phết",
    ],
    "central": [
        "chi rứa hè", "chi rứa", "chi mô", "mần chi", "răng rứa", "răng ri", "mần răng",
        "đi mô rứa", "ở mô rứa", "mô rồi", "răng rồi", "rứa hè", "rứa hầy", "nớ hè", "ri hè",
        "rứa", "nớ", "ni", "mần", "trốc", "nác", "tề", "chừ", "hầy",
        "choa", "bọn choa", "mệ", "nờ", "nỏ", "gấy", "nẫu", "túi ni", "hí",
    ],
    "southern": [
        "hổng", "hông dzậy", "hông", "nghen", "à nghen", "dạ hen", "hen",
        "má ơi", "trời đất ơi", "chèn ơi", "mèn ơi", "quá trời", "dữ hông", "dữ vậy", "dữ dzậy",
        "tui", "dìa", "dzậy", "dzô", "biết chớ", "chớ bộ", "đây nè", "bây giờ nè", "nè",
        "nói dzậy", "quơ", "trỏng", "quẹo", "được hôn", "vậy hôn", "hôn ta", "mắc quá", "mắc dữ",
    ],
}

AMBIGUOUS_CONTEXT_BLACKLIST = {
    "ni": ["ni cô", "ni sư", "ni-lông", "nilon"],
    "tề": ["chỉnh tề"],
    "nỏ": ["cây nỏ", "bắn nỏ", "mũi tên nỏ"],
    "hí": ["hí hửng", "hí hoáy", "hí hí"],
    "nè": [],
}


def _compile_patterns(lexicon):
    compiled = {}
    for region, words in lexicon.items():
        words_sorted = sorted(set(words), key=len, reverse=True)
        pattern = r"(?<!\w)(" + "|".join(re.escape(w) for w in words_sorted) + r")(?!\w)"
        compiled[region] = re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)
    return compiled


REGION_PATTERNS = _compile_patterns(REGION_LEXICON)

GEO_PROVINCE_MAP = {
    "hà nội": "northern", "hải phòng": "northern", "bắc ninh": "northern",
    "nam định": "northern", "thái bình": "northern", "ninh bình": "northern",
    "hải dương": "northern", "hưng yên": "northern", "vĩnh phúc": "northern",
    "phú thọ": "northern", "quảng ninh": "northern", "lào cai": "northern",
    "yên bái": "northern", "điện biên": "northern", "sơn la": "northern",
    "hòa bình": "northern", "thái nguyên": "northern", "bắc giang": "northern",
    "lạng sơn": "northern", "cao bằng": "northern", "tuyên quang": "northern",
    "hà giang": "northern", "bắc kạn": "northern", "lai châu": "northern",

    "thanh hóa": "central", "nghệ an": "central", "hà tĩnh": "central",
    "quảng bình": "central", "quảng trị": "central", "huế": "central",
    "thừa thiên huế": "central", "đà nẵng": "central", "quảng nam": "central",
    "quảng ngãi": "central", "bình định": "central", "phú yên": "central",
    "khánh hòa": "central", "kon tum": "central", "gia lai": "central",
    "đắk lắk": "central", "đắk nông": "central", "ninh thuận": "central",
    "bình thuận": "central", "lâm đồng": "central",

    "tp hcm": "southern", "hồ chí minh": "southern", "sài gòn": "southern",
    "bà rịa": "southern", "vũng tàu": "southern", "đồng nai": "southern",
    "bình dương": "southern", "bình phước": "southern", "tây ninh": "southern",
    "long an": "southern", "tiền giang": "southern", "bến tre": "southern",
    "trà vinh": "southern", "vĩnh long": "southern", "đồng tháp": "southern",
    "an giang": "southern", "kiên giang": "southern", "cần thơ": "southern",
    "hậu giang": "southern", "sóc trăng": "southern", "bạc liêu": "southern",
    "cà mau": "southern",
}


def _strip_accents_lower(text):
    return unicodedata.normalize("NFC", text).strip().lower()


def geo_lookup_region(location):
    if not location or not isinstance(location, str):
        return None
    loc = _strip_accents_lower(location)
    for province, region in GEO_PROVINCE_MAP.items():
        if province in loc:
            return region
    return None


def _is_blacklisted(text_lower, matched_word):
    blocked_phrases = AMBIGUOUS_CONTEXT_BLACKLIST.get(matched_word.lower(), [])
    return any(phrase in text_lower for phrase in blocked_phrases)


def get_dialect_features(text):
    if not text:
        return {}
    text_lower = text.lower()
    features = {}
    for region, pattern in REGION_PATTERNS.items():
        matches = pattern.findall(text)
        features[region] = [m for m in matches if not _is_blacklisted(text_lower, m)]
    return features


def lexicon_lookup_region(text):
    features = get_dialect_features(text)
    scores = {region: len(words) for region, words in features.items()}
    nonzero = {r: s for r, s in scores.items() if s > 0}
    if not nonzero:
        return None, scores
    best_region = max(nonzero, key=nonzero.get)
    return best_region, scores


def assign_region(text, location=None):
    geo_region = geo_lookup_region(location)
    if geo_region is not None:
        return {"region": geo_region, "label_source": "geo_metadata", "confidence": "high", "match_scores": None}

    lex_region, scores = lexicon_lookup_region(text)
    if lex_region is not None:
        return {"region": lex_region, "label_source": "lexicon", "confidence": "medium", "match_scores": scores}

    return {"region": None, "label_source": "unmatched", "confidence": "none", "match_scores": scores}


if __name__ == "__main__":
    demo_samples = [
        ("Mi đi mô rứa hè?", None),
        ("Tui hổng biết nghen, dữ vậy trời!", None),
        ("Hôm nay trời đẹp ghê, đi chơi không?", "Sài Gòn"),
        ("Mô típ giống vài bạn ai đồ tóp tóp", None),
        ("Bán cà phê vỉa hè mà bận đồ đó chẳng lịch sự chút nào", None),
        ("Sao bịt khẩu trang qoài z mở ra cái răng hô mưa không dột kk", None),
    ]
    for text, loc in demo_samples:
        result = assign_region(text, location=loc)
        print(f'"{text}" (location={loc}) -> {result["region"]} '
              f'[nguồn: {result["label_source"]}, tin cậy: {result["confidence"]}]')
