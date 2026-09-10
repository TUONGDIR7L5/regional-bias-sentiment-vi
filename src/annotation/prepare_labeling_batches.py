from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
IN_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup.csv"
INDEXED_PATH = ROOT_DIR / "data" / "interim" / "merged_dedup_indexed.csv"
ANNOTATOR_1_OUT = ROOT_DIR / "data" / "annotations" / "annotator_1" / "to_label.csv"
ANNOTATOR_2_OUT = ROOT_DIR / "data" / "annotations" / "annotator_2" / "to_label.csv"

DOUBLE_CHECK_RATIO = 0.15
SEED = 42


def extract():
    df = pd.read_csv(IN_PATH)
    df["row_id"] = range(len(df))
    print(f"[EXTRACT] Đọc {len(df)} dòng, gán row_id")
    return df


def transform(df):
    df["sentiment"] = df["sentiment"].fillna("").astype(str)
    need_label = df[df["sentiment"].str.strip() == ""].copy()
    already_labeled = df[df["sentiment"].str.strip() != ""].copy()
    print(f"[TRANSFORM] {len(already_labeled)} dòng đã có nhãn sẵn "
          f"(vidia2std test + synthetic), {len(need_label)} dòng cần gán nhãn thủ công")

    double_check_parts = [g.sample(frac=DOUBLE_CHECK_RATIO, random_state=SEED)
                           for _, g in need_label.groupby("region")]
    double_check = pd.concat(double_check_parts)
    print(f"[TRANSFORM] Chọn {len(double_check)} dòng (~{DOUBLE_CHECK_RATIO:.0%}) "
          f"để gán nhãn đôi phục vụ đo Cohen's Kappa")
    return df, need_label, double_check


def load(df, need_label, double_check):
    INDEXED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEXED_PATH, index=False, encoding="utf-8")

    ANNOTATOR_1_OUT.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATOR_2_OUT.parent.mkdir(parents=True, exist_ok=True)

    a1 = need_label[["row_id", "dialect", "region"]].copy()
    a1["sentiment"] = ""
    a1.to_csv(ANNOTATOR_1_OUT, index=False, encoding="utf-8")

    a2 = double_check[["row_id", "dialect", "region"]].copy()
    a2["sentiment"] = ""
    a2.to_csv(ANNOTATOR_2_OUT, index=False, encoding="utf-8")

    print(f"[LOAD] Đã ghi {len(a1)} dòng vào {ANNOTATOR_1_OUT} (gán nhãn toàn bộ)")
    print(f"[LOAD] Đã ghi {len(a2)} dòng vào {ANNOTATOR_2_OUT} (gán nhãn đôi, kiểm tra Kappa)")
    print("[LOAD] Điền cột sentiment (positive/negative/neutral) vào 2 file trên rồi lưu lại, "
          "sau đó chạy src/annotation/agreement_kappa.py để gộp nhãn.")


def run_etl_pipeline():
    df = extract()
    df, need_label, double_check = transform(df)
    load(df, need_label, double_check)
    return df


if __name__ == "__main__":
    run_etl_pipeline()
