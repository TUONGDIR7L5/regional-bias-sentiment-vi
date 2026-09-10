import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HF_BASE = "https://huggingface.co/datasets/Biu3010/ViDia2Std/resolve/main"
SPLITS = ["train", "dev", "test"]
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "vidia2std"


def extract():
    dfs = {}
    for split in SPLITS:
        url = f"{HF_BASE}/{split}.csv"
        df = pd.read_csv(url)
        print(f"[EXTRACT] {split}: {len(df)} dong, cot {list(df.columns)}")
        dfs[split] = df
    return dfs


def transform(dfs):
    for split, df in dfs.items():
        df = df.dropna(subset=["dialect", "standard", "region"]).copy()
        if "sentiment" not in df.columns:
            df["sentiment"] = None
        df["source"] = "vidia2std"
        df["source_split"] = split
        dfs[split] = df[["dialect", "standard", "region", "sentiment", "source", "source_split"]]
    return dfs


def load(dfs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, df in dfs.items():
        out_path = OUT_DIR / f"{split}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[LOAD] Da ghi {len(df)} dong vao {out_path}")


def run_etl_pipeline():
    dfs = extract()
    dfs = transform(dfs)
    load(dfs)
    return dfs


if __name__ == "__main__":
    run_etl_pipeline()
