import re
import unicodedata

REGION_LEXICON = {
    "northern": [
        "giời", "u ơi", "bu ơi", "chả biết", "chúng mày", "đấy nhở", "nhỉ",
        "cơ mà", "thế cơ đấy", "vãi", "ối giời ơi",
    ],
    "central": [
        "mô", "tê", "răng", "rứa", "chi rứa", "nớ", "hè", "ni", "mần",
        "trốc", "nác", "tề", "chừ", "ri", "hí", "mụ", "dừ",
    ],
    "southern": [
        "hổng", "nghen", "dữ", "hông", "má ơi", "trời đất", "dạ hen",
        "hen", "mắc", "tui", "hôn", "chèn ơi", "quá trời", "dữ hông",
        "à nghen", "nè",
    ],
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
    # Bac Bo
    "hà nội": "northern", "hải phòng": "northern", "bắc ninh": "northern",
    "nam định": "northern", "thái bình": "northern", "ninh bình": "northern",
    "hải dương": "northern", "hưng yên": "northern", "vĩnh phúc": "northern",
    "phú thọ": "northern", "quảng ninh": "northern", "lào cai": "northern",
    "yên bái": "northern", "điện biên": "northern", "sơn la": "northern",
    "hòa bình": "northern", "thái nguyên": "northern", "bắc giang": "northern",
    "lạng sơn": "northern", "cao bằng": "northern", "tuyên quang": "northern",
    "hà giang": "northern", "bắc kạn": "northern", "lai châu": "northern",

    # Bac Trung Bo & Nam Trung Bo & Tay Nguyen
    "thanh hóa": "central", "nghệ an": "central", "hà tĩnh": "central",
    "quảng bình": "central", "quảng trị": "central", "huế": "central",
    "thừa thiên huế": "central", "đà nẵng": "central", "quảng nam": "central",
    "quảng ngãi": "central", "bình định": "central", "phú yên": "central",
    "khánh hòa": "central", "kon tum": "central", "gia lai": "central",
    "đắk lắk": "central", "đắk nông": "central", "ninh thuận": "central",
    "bình thuận": "central", "lâm đồng": "central",

    # Nam Bo
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


def lexicon_lookup_region(text):
    scores = {region: len(pattern.findall(text)) for region, pattern in REGION_PATTERNS.items()}
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

    return {"region": "northern", "label_source": "fallback_default", "confidence": "low", "match_scores": scores}


if __name__ == "__main__":
    demo_samples = [
        ("Mi đi mô rứa hè?", None),
        ("Tui hổng biết nghen, dữ vậy trời!", None),
        ("Hôm nay trời đẹp ghê, đi chơi không?", "Sài Gòn"),
        ("Bài viết này chất lượng đấy nhỉ.", None),
    ]
    for text, loc in demo_samples:
        result = assign_region(text, location=loc)
        print(f'"{text}" (location={loc}) -> {result["region"]} '
              f'[nguồn: {result["label_source"]}, tin cậy: {result["confidence"]}]')
