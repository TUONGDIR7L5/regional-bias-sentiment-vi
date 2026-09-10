import json
import re
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.geo_metadata_labeling import REGION_LEXICON

ROOT_DIR = Path(__file__).resolve().parents[2]
IN_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup_labeled.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "regional_dictionary.json"

SENTIMENT_WORDS = {
    "tuyệt vời", "ngon", "xịn", "hay", "đẹp", "đã", "thích", "mê", "ưng",
    "tệ", "chán", "dở", "xấu", "kém", "buồn", "tức", "sợ", "lo",
}


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def extract():
    df = pd.read_csv(IN_PATH)
    df = df.dropna(subset=["standard"])
    df = df[df["standard"].str.strip() != ""]
    print(f"[EXTRACT] {len(df)} dòng có cặp dialect-standard để căn chỉnh")
    return df


def align_pair(dialect, standard):
    dia_tokens = tokenize(dialect)
    std_tokens = tokenize(standard)
    if len(dia_tokens) != len(std_tokens):
        return []
    return [(d, s) for d, s in zip(dia_tokens, std_tokens) if d != s]


def transform(df):
    mapping = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        pairs = align_pair(row["dialect"], row["standard"])
        for dia_word, std_word in pairs:
            mapping[row["region"]][(dia_word, std_word)] += 1

    dictionary = {}
    for region, seed_words in REGION_LEXICON.items():
        entries = []
        for word in seed_words:
            entries.append({
                "dialect_word": word,
                "standard_word": None,
                "sentiment_bearing": word in SENTIMENT_WORDS,
                "frequency": 0,
            })
        for (dia_word, std_word), freq in mapping.get(region, {}).items():
            entries.append({
                "dialect_word": dia_word,
                "standard_word": std_word,
                "sentiment_bearing": dia_word in SENTIMENT_WORDS or std_word in SENTIMENT_WORDS,
                "frequency": freq,
            })
        dictionary[region] = entries
    print(f"[TRANSFORM] Xây dựng từ điển cho {len(dictionary)} vùng miền")
    return dictionary


def load(dictionary):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print(f"[LOAD] Đã ghi từ điển phương ngữ vào {OUT_PATH}")


def run_etl_pipeline():
    df = extract()
    dictionary = transform(df)
    load(dictionary)
    return dictionary


if __name__ == "__main__":
    run_etl_pipeline()
