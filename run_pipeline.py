import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_collection import fetch_vidia2std
from src.data_collection.crawlers import social_media_crawler
from src.data_collection.synthetic_generation import generate_from_vidia2std
from src.preprocessing import merge_and_dedup, dialect_feature_extractor, stratified_split, compare_datasets
from src.annotation import prepare_labeling_batches, agreement_kappa
from src.dictionary import build_regional_dictionary

STAGES = [
    ("fetch_vidia2std", fetch_vidia2std.run_etl_pipeline),
    ("crawl_youtube", social_media_crawler.run_etl_pipeline),
    ("generate_synthetic", generate_from_vidia2std.run_etl_pipeline),
    ("merge_and_dedup", merge_and_dedup.run_etl_pipeline),
    ("prepare_labeling_batches", prepare_labeling_batches.run_etl_pipeline),
    ("finalize_labels", agreement_kappa.run_etl_pipeline),
    ("build_regional_dictionary", build_regional_dictionary.run_etl_pipeline),
    ("dialect_feature_extractor", dialect_feature_extractor.run_etl_pipeline),
    ("stratified_split", stratified_split.run_etl_pipeline),
    ("compare_datasets", compare_datasets.run_etl_pipeline),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    for name, fn in STAGES:
        if args.only and name != args.only:
            continue
        print(f"\n===== STAGE: {name} =====")
        if name == "finalize_labels":
            a1 = Path("data/annotations/annotator_1/to_label.csv")
            if a1.exists():
                import pandas as pd
                filled = pd.read_csv(a1)["sentiment"].fillna("").astype(str).str.strip().ne("").all()
                if not filled:
                    print("[STAGE] Chưa gán nhãn xong ở data/annotations/annotator_1/to_label.csv, dừng lại.")
                    break
        fn()


if __name__ == "__main__":
    main()
