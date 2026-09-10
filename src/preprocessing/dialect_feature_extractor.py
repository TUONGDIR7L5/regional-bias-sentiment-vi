import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.geo_metadata_labeling import get_dialect_features

ROOT_DIR = Path(__file__).resolve().parents[2]
IN_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup_labeled.csv"
OUT_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup_labeled_features.csv"


def extract():
    df = pd.read_csv(IN_PATH)
    print(f"[EXTRACT] Doc {len(df)} dong")
    return df


def transform(df):
    def features_for_row(row):
        matches = get_dialect_features(row["dialect"])
        own_region_words = matches.get(row["region"], [])
        return json.dumps(sorted(set(own_region_words)), ensure_ascii=False)

    df["dialect_features"] = df.apply(features_for_row, axis=1)
    n_with_features = (df["dialect_features"] != "[]").sum()
    print(f"[TRANSFORM] {n_with_features}/{len(df)} dòng có ít nhất 1 từ đặc trưng vùng miền")
    return df


def load(df):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"[LOAD] Đã ghi {len(df)} dòng vào {OUT_PATH}")


def run_etl_pipeline():
    df = extract()
    df = transform(df)
    load(df)
    return df


if __name__ == "__main__":
    run_etl_pipeline()
