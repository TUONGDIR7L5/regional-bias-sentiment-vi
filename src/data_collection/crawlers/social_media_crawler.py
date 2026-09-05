import csv
import os
import time

import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_collection.geo_metadata_labeling import assign_region
from src.preprocessing.cleaning import clean_text

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

REGION_KEYWORDS = {
    "northern": [
        "giọng Bắc chuẩn", "nói tiếng Hà Nội gốc", "phỏng vấn người Hà Nội",
        "giọng Hải Phòng", "giọng Nam Định", "hài giọng Bắc",
        "vlog người Hà Nội xưa", "ông bà nói chuyện Hà Nội",
        "tiếng lóng Hà Nội", "giọng Bắc Ninh",
    ],
    "central": [
        "mô tê răng rứa", "giọng Nghệ An", "giọng Hà Tĩnh",
        "phỏng vấn tiếng Huế", "giọng Quảng Nam", "giọng Quảng Trị",
        "giọng Quảng Ngãi", "hài giọng miền Trung", "test giọng Huế",
        "nói chuyện Nghệ Tĩnh", "vlog người Đà Nẵng", "giọng Bình Định",
    ],
    "southern": [
        "hổng nghen dữ", "giọng miền Tây", "phỏng vấn Sài Gòn xưa",
        "giọng Sài Gòn gốc", "hài giọng miền Tây", "test giọng miền Nam",
        "vlog người Cần Thơ", "giọng An Giang", "nói chuyện lục tỉnh",
        "giọng Bến Tre",
    ],
}

VIDEOS_PER_KEYWORD = 5     # search ton 100 unit/lan goi -> gioi han so luong
COMMENTS_PER_VIDEO = 100   # toi da 100 comment/lan goi commentThreads


def search_video_ids(keyword, max_results=VIDEOS_PER_KEYWORD):
    params = {
        "part": "id",
        "q": keyword,
        "type": "video",
        "maxResults": max_results,
        "relevanceLanguage": "vi",
        "key": YOUTUBE_API_KEY,
    }
    resp = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    return [item["id"]["videoId"] for item in resp.json().get("items", [])]


def fetch_comments(video_id, max_results=COMMENTS_PER_VIDEO):
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY,
    }
    resp = requests.get(YOUTUBE_COMMENTS_URL, params=params, timeout=10)
    if resp.status_code != 200:
        return []
    records = []
    for item in resp.json().get("items", []):
        top = item["snippet"]["topLevelComment"]["snippet"]
        records.append({
            "text": top["textDisplay"],
            "source": f"youtube:{video_id}",
            "location": None,  # YouTube khong tra ve dia diem nguoi binh luan
            "crawl_time": top["publishedAt"],
        })
    return records


def crawl_raw_data():
    if not YOUTUBE_API_KEY:
        print("[EXTRACT] Chưa set biến môi trường YOUTUBE_API_KEY, bỏ qua crawl.")
        return []

    all_records = []
    for region, keywords in REGION_KEYWORDS.items():
        for keyword in keywords:
            video_ids = search_video_ids(keyword)
            print(f"[EXTRACT] '{keyword}' ({region}): tìm được {len(video_ids)} video")
            for video_id in video_ids:
                comments = fetch_comments(video_id)
                all_records.extend(comments)
                time.sleep(0.2)  # tránh gọi API dồn dập
    print(f"[EXTRACT] Tổng cộng thu được {len(all_records)} bình luận thô")
    return all_records


MIN_LEXICON_SCORE = 2  # phải khớp ít nhất 2 từ đặc trưng mới coi là đáng tin


def transform_records(records):
    rows = []
    for r in records:
        cleaned = clean_text(r.get("text"))
        if not cleaned:
            continue
        label = assign_region(cleaned, location=r.get("location"))

        if label["label_source"] != "lexicon":
            continue

        best_score = label["match_scores"][label["region"]]
        if best_score < MIN_LEXICON_SCORE:
            continue

        rows.append({
            "text": cleaned,
            "source": r.get("source", "unknown"),
            "region": label["region"],
            "label_source": label["label_source"],
            "confidence": label["confidence"],
            "match_score": best_score,
        })
    return rows


def load_to_storage(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[LOAD] Không có dòng nào để ghi vào {out_path}")
        return
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[LOAD] Đã ghi {len(rows)} dòng vào {out_path}")


DEFAULT_OUT_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "crawled" / "social_media" / "crawled_labeled.csv"

def run_etl_pipeline(out_path=DEFAULT_OUT_PATH):
    raw = crawl_raw_data()
    rows = transform_records(raw)
    load_to_storage(rows, out_path)
    return rows


if __name__ == "__main__":
    run_etl_pipeline()
