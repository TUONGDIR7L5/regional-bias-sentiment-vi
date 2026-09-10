from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT_DIR = Path(__file__).resolve().parents[2]
INDEXED_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup_indexed.csv"
ANNOTATOR_1_PATH = ROOT_DIR / "data" / "annotations" / "annotator_1" / "to_label.csv"
ANNOTATOR_2_PATH = ROOT_DIR / "data" / "annotations" / "annotator_2" / "to_label.csv"
KAPPA_OUT = ROOT_DIR / "data" / "annotations" / "agreement_reports" / "kappa_report.csv"
OUT_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup_labeled.csv"

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def extract():
    df = pd.read_csv(INDEXED_PATH)
    a1 = pd.read_csv(ANNOTATOR_1_PATH)
    a2 = pd.read_csv(ANNOTATOR_2_PATH)
    print(f"[EXTRACT] {len(a1)} nhãn từ annotator_1, {len(a2)} nhãn từ annotator_2 (tập chồng lấn)")
    return df, a1, a2


def compute_kappa(a1, a2):
    merged = a1.merge(a2, on="row_id", suffixes=("_a1", "_a2"))
    merged = merged.dropna(subset=["sentiment_a1", "sentiment_a2"])
    merged = merged[merged["sentiment_a1"].isin(VALID_SENTIMENTS) & merged["sentiment_a2"].isin(VALID_SENTIMENTS)]

    rows = []
    if len(merged) > 0:
        overall = cohen_kappa_score(merged["sentiment_a1"], merged["sentiment_a2"])
        rows.append({"region": "ALL", "n_samples": len(merged), "cohen_kappa": round(overall, 4)})
        for region, group in merged.groupby("region_a1"):
            k = cohen_kappa_score(group["sentiment_a1"], group["sentiment_a2"])
            rows.append({"region": region, "n_samples": len(group), "cohen_kappa": round(k, 4)})
    return merged, pd.DataFrame(rows)


def transform(df, a1, a2):
    merged_overlap, kappa_df = compute_kappa(a1, a2)
    print(f"[TRANSFORM] Báo cáo Cohen's Kappa:\n{kappa_df}")

    final_labels = dict(zip(a1["row_id"], a1["sentiment"]))
    agree_ids = set(merged_overlap[merged_overlap["sentiment_a1"] == merged_overlap["sentiment_a2"]]["row_id"])
    disagree_ids = set(merged_overlap["row_id"]) - agree_ids
    print(f"[TRANSFORM] {len(agree_ids)} dòng 2 người đồng ý, {len(disagree_ids)} dòng cần xem lại thủ công")

    df["sentiment"] = df["sentiment"].fillna("").astype(str)

    def resolve(row):
        if row["sentiment"].strip() != "":
            return row["sentiment"]
        return final_labels.get(row["row_id"], "")

    df["sentiment"] = df.apply(resolve, axis=1)
    df["needs_review"] = df["row_id"].isin(disagree_ids)
    df = df[df["sentiment"].isin(VALID_SENTIMENTS)]
    print(f"[TRANSFORM] Tổng {len(df)} dòng có nhãn cảm xúc hợp lệ sau khi gộp")
    return df, kappa_df


def load(df, kappa_df):
    KAPPA_OUT.parent.mkdir(parents=True, exist_ok=True)
    kappa_df.to_csv(KAPPA_OUT, index=False, encoding="utf-8")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"[LOAD] Đã ghi {len(df)} dòng vào {OUT_PATH}")
    print(f"[LOAD] Đã ghi báo cáo Kappa vào {KAPPA_OUT}")


def run_etl_pipeline():
    df, a1, a2 = extract()
    df, kappa_df = transform(df, a1, a2)
    load(df, kappa_df)
    return df


if __name__ == "__main__":
    run_etl_pipeline()
