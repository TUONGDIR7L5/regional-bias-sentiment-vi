import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.preprocessing.cleaning import clean_text, is_valid_sentence

ROOT_DIR = Path(__file__).resolve().parents[2]
VIDIA2STD_DIR = ROOT_DIR / "data" / "raw" / "vidia2std"
CRAWLED_PATH = ROOT_DIR / "data" / "raw" / "crawled" / "social_media" / "crawled.csv"
SYNTHETIC_PATH = ROOT_DIR / "data" / "raw" / "synthetic" / "synthetic.csv"
OUT_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup.csv"

COLUMNS = ["dialect", "standard", "region", "sentiment", "source", "source_split"]


def source_priority(source):
    if source == "vidia2std":
        return 0
    if isinstance(source, str) and source.startswith("youtube"):
        return 1
    if source == "synthetic":
        return 2
    return 9


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[\"'\u2018\u2019\u201c\u201d]", "", s)
    s = re.sub(r"[.,!?;:…]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def extract():
    frames = []
    for split in ("train", "dev", "test"):
        path = VIDIA2STD_DIR / f"{split}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if CRAWLED_PATH.exists():
        frames.append(pd.read_csv(CRAWLED_PATH))
    if SYNTHETIC_PATH.exists():
        frames.append(pd.read_csv(SYNTHETIC_PATH))
    if not frames:
        print("[EXTRACT] Không tìm thấy nguồn dữ liệu raw nào.")
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True, sort=False)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    print(f"[EXTRACT] Gộp {len(frames)} nguồn -> {len(df)} dòng thô")
    return df[COLUMNS]


def transform(df):
    df["dialect"] = df["dialect"].apply(clean_text)
    df["standard"] = df["standard"].apply(clean_text)
    df = df[df["dialect"].apply(lambda t: is_valid_sentence(t))]
    df = df.dropna(subset=["region"])
    df = df[df["region"].isin(["northern", "central", "southern"])]

    df["_n_dia"] = df["dialect"].apply(norm)
    df["_n_std"] = df["standard"].apply(norm)
    df["_priority"] = df["source"].apply(source_priority)
    df = df.sort_values("_priority")

    df = df.drop_duplicates(subset="_n_dia", keep="first")
    df["_dedup_key"] = df["_n_std"].where(df["_n_std"] != "", df["_n_dia"])
    df = df.drop_duplicates(subset="_dedup_key", keep="first")

    df = df.drop(columns=["_n_dia", "_n_std", "_dedup_key", "_priority"])
    print(f"[TRANSFORM] Sau khi lọc + khử trùng lặp -> {len(df)} dòng")
    return df


def load(df):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"[LOAD] Đã ghi {len(df)} dòng vào {OUT_PATH}")
    print(df.groupby(["region", "source"]).size())


def run_etl_pipeline():
    df = extract()
    df = transform(df)
    load(df)
    return df


if __name__ == "__main__":
    run_etl_pipeline()
