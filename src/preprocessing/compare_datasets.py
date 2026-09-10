import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_collection.geo_metadata_labeling import get_dialect_features

ROOT_DIR = Path(__file__).resolve().parents[2]
OUR_DATA_PATHS = [ROOT_DIR / "data" / "processed" / f"{s}.csv" for s in ("train", "dev", "test")]
PUBLIC_DIR = ROOT_DIR / "data" / "raw" / "public_datasets"
OUT_PATH = ROOT_DIR / "reports" / "dataset_comparison_report.md"


def extract():
    our_df = pd.concat([pd.read_csv(p) for p in OUR_DATA_PATHS if p.exists()], ignore_index=True)
    public_dfs = {}
    for path in PUBLIC_DIR.glob("*.csv"):
        public_dfs[path.stem] = pd.read_csv(path)
    print(f"[EXTRACT] Bộ dữ liệu mới: {len(our_df)} dòng, {len(public_dfs)} bộ dữ liệu công khai để so sánh")
    return our_df, public_dfs


def dialect_density(texts):
    total_words, total_matches = 0, 0
    for text in texts:
        total_words += len(str(text).split())
        matches = get_dialect_features(str(text))
        total_matches += sum(len(v) for v in matches.values())
    return round(total_matches / max(total_words, 1), 4)


def transform(our_df, public_dfs):
    stats = []
    stats.append({
        "dataset": "Bộ dữ liệu mới (đề tài)",
        "n_samples": len(our_df),
        "region_balance": our_df["region"].value_counts(normalize=True).round(3).to_dict(),
        "sentiment_balance": our_df["sentiment"].value_counts(normalize=True).round(3).to_dict()
        if "sentiment" in our_df.columns else None,
        "dialect_word_density": dialect_density(our_df["dialect"]),
    })
    for name, df in public_dfs.items():
        text_col = next((c for c in df.columns if c.lower() in ("text", "sentence", "content")), df.columns[0])
        stats.append({
            "dataset": name,
            "n_samples": len(df),
            "region_balance": None,
            "sentiment_balance": None,
            "dialect_word_density": dialect_density(df[text_col]),
        })
    return stats


def load(stats):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# So sánh định lượng với các bộ dữ liệu hiện có\n"]
    lines.append("| Dataset | Số mẫu | Mật độ từ phương ngữ | Cân bằng vùng miền | Cân bằng cảm xúc |")
    lines.append("|---|---|---|---|---|")
    for s in stats:
        lines.append(
            f"| {s['dataset']} | {s['n_samples']} | {s['dialect_word_density']} | "
            f"{s['region_balance']} | {s['sentiment_balance']} |"
        )
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[LOAD] Đã ghi báo cáo so sánh vào {OUT_PATH}")


def run_etl_pipeline():
    our_df, public_dfs = extract()
    stats = transform(our_df, public_dfs)
    load(stats)
    return stats


if __name__ == "__main__":
    run_etl_pipeline()
