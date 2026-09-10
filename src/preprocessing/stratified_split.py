from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
IN_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup_labeled.csv"
OUT_DIR = ROOT_DIR / "data" / "processed"

TARGET_TOTAL = 40000
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SEED = 42


def extract():
    df = pd.read_csv(IN_PATH)
    df = df.dropna(subset=["sentiment"])
    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]
    print(f"[EXTRACT] Doc {len(df)} dong da gan nhan cam xuc")
    return df


def transform(df):
    if len(df) > TARGET_TOTAL:
        df, _ = train_test_split(
            df, train_size=TARGET_TOTAL, random_state=SEED,
            stratify=df["region"] + "_" + df["sentiment"],
        )
        print(f"[TRANSFORM] Lay mau xuong con {len(df)} dong theo muc tieu {TARGET_TOTAL}")

    strat_key = df["region"] + "_" + df["sentiment"]
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_RATIO, random_state=SEED, stratify=strat_key,
    )
    strat_key_temp = temp_df["region"] + "_" + temp_df["sentiment"]
    dev_ratio_in_temp = DEV_RATIO / (DEV_RATIO + TEST_RATIO)
    dev_df, test_df = train_test_split(
        temp_df, train_size=dev_ratio_in_temp, random_state=SEED, stratify=strat_key_temp,
    )
    print(f"[TRANSFORM] train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")
    return {"train": train_df, "dev": dev_df, "test": test_df}


def load(splits):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        out_path = OUT_DIR / f"{name}.csv"
        split_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[LOAD] Đã ghi {len(split_df)} dòng vào {out_path}")
        print(split_df.groupby(["region", "sentiment"]).size())


def run_etl_pipeline():
    df = extract()
    splits = transform(df)
    load(splits)
    return splits


if __name__ == "__main__":
    run_etl_pipeline()
