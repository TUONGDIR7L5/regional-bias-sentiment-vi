import csv
import json
import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import requests

from src.data_collection.geo_metadata_labeling import assign_region
from src.preprocessing.cleaning import clean_text, expand_abbreviations, is_valid_sentence

YOUTUBE_API_KEYS = [k.strip().strip('"').strip("'") for k in os.getenv("YOUTUBE_API_KEYS", "").split(",") if k.strip()]
if not YOUTUBE_API_KEYS and os.getenv("YOUTUBE_API_KEY"):
    YOUTUBE_API_KEYS = [os.getenv("YOUTUBE_API_KEY").strip().strip('"').strip("'")]

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

BAIT_KEYWORDS = [
    "đoán giọng vùng miền", "thử thách giọng ba miền", "so sánh giọng 3 miền",
    "từ địa phương ba miền", "phương ngữ tiếng Việt", "người miền nào",
    "test giọng vùng miền", "giọng quê bạn", "đố bạn giọng miền nào",
    "phân biệt giọng bắc trung nam", "người bắc học giọng nam",
    "người nam nói giọng bắc", "từ ngữ mỗi miền khác nhau",
    "người xa quê tâm sự", "hỏi quê bạn ở đâu",
    "combat giọng vùng miền", "học giọng miền khác cho vui",
    "chê giọng vùng miền", "cà khịa giọng vùng miền", "quê bạn ở đâu vậy",
    "nghe giọng đoán quê", "ghép đôi theo giọng nói",
]

REGION_KEYWORDS_SEED = {
    "northern": [
        "vlog cuộc sống nông thôn miền Bắc", "mâm cơm gia đình miền Bắc",
        "chợ quê Bắc Bộ", "cuộc sống thường ngày Hà Nội",
        "vlog sinh viên Hà Nội", "đi chợ Hà Nội", "nấu ăn miền Bắc",
        "làm vườn miền Bắc", "cuộc sống Thái Bình", "vlog Nam Định",
        "review quán ăn Hà Nội", "một ngày của mẹ bỉm miền Bắc",
        "cuộc sống Hải Phòng", "chuyện làng quê Bắc Bộ",
        "vlog Bắc Ninh", "vlog Ninh Bình", "vlog Hải Dương", "vlog Thanh Hóa",
        "một ngày làm nông dân miền Bắc", "vlog về quê ăn Tết miền Bắc",
        "cuộc sống công nhân miền Bắc", "đám cưới quê Bắc Bộ",
        "phiên chợ Tết miền Bắc", "review đồ ăn vặt Hà Nội",
        "vlog xóm làng miền Bắc", "cuộc sống Sơn La", "cuộc sống Phú Thọ",
    ],
    "central": [
        "vlog cuộc sống Nghệ An", "mâm cơm gia đình miền Trung",
        "chợ quê Hà Tĩnh", "cuộc sống thường ngày ở Huế",
        "vlog nông thôn miền Trung", "đi chợ Huế", "nấu ăn miền Trung",
        "làm ruộng Nghệ An", "vlog Quảng Ngãi", "cuộc sống Bình Định",
        "review quán ăn Đà Nẵng", "chuyện làng quê xứ Nghệ",
        "vlog Quảng Nam", "cuộc sống Quảng Trị",
        "vlog Quảng Bình", "vlog Phú Yên", "vlog Khánh Hòa", "cuộc sống Thanh Hóa",
        "một ngày làm nông dân miền Trung", "vlog về quê ăn Tết miền Trung",
        "cuộc sống công nhân miền Trung", "đám cưới quê miền Trung",
        "phiên chợ Tết miền Trung", "review đồ ăn vặt Đà Nẵng",
        "vlog xóm làng miền Trung", "cuộc sống Kon Tum", "cuộc sống Gia Lai",
        "vlog người dân Nghệ Tĩnh", "chuyện nhà nông xứ Huế",
    ],
    "southern": [
        "vlog cuộc sống miền Tây", "mâm cơm gia đình miền Tây",
        "chợ quê miền Tây", "cuộc sống thường ngày Sài Gòn",
        "đi chợ Cần Thơ", "nấu ăn miền Tây", "làm vườn miền Tây",
        "vlog nông thôn miền Nam", "cuộc sống Bến Tre",
        "review quán ăn Sài Gòn", "một ngày ở quê miền Tây",
        "vlog An Giang", "cuộc sống Cà Mau", "chuyện xóm miền Tây",
    ],
}


def build_region_keywords():
    keywords = {"__general__": BAIT_KEYWORDS}
    keywords.update(REGION_KEYWORDS_SEED)
    return keywords


REGION_KEYWORDS = build_region_keywords()

VIDEOS_PER_KEYWORD = 50
COMMENT_PAGES_PER_VIDEO = 3
COMMENTS_PER_PAGE = 100
MAX_RETRIES = 2
MIN_LEXICON_SCORE = 1

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "crawled" / "social_media" / "_state.json"
RAW_OUT_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "crawled" / "social_media" / "raw_comments.csv"
OUT_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "crawled" / "social_media" / "crawled.csv"


class QuotaExhausted(Exception):
    pass


class KeyPool:
    def __init__(self, keys):
        self.keys = keys
        self.idx = 0

    def current(self):
        if not self.keys:
            return None
        return self.keys[self.idx]

    def rotate(self):
        self.idx += 1
        return self.idx < len(self.keys)


QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded"}


def _error_reason(resp):
    try:
        return resp.json()["error"]["errors"][0]["reason"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _get_with_retry(url, params, key_pool):
    params = dict(params)
    for attempt in range(MAX_RETRIES):
        key = key_pool.current()
        if key is None:
            raise QuotaExhausted("Hết API key khả dụng")
        params["key"] = key
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 403:
            reason = _error_reason(resp)
            if reason in QUOTA_REASONS:
                print(f"[EXTRACT] Key hiện tại hết quota ({reason}), chuyển key kế tiếp.")
                if not key_pool.rotate():
                    raise QuotaExhausted("Tất cả API key đều hết quota")
                continue
            print(f"[EXTRACT] Bỏ qua (403 - {reason}): {resp.text[:200]}")
            return None
        if resp.status_code == 429:
            print(f"[EXTRACT] Rate limit (429), chuyển key kế tiếp.")
            if not key_pool.rotate():
                raise QuotaExhausted("Tất cả API key đều hết quota")
            continue
        print(f"[EXTRACT] Loi API {resp.status_code}: {resp.text[:300]}")
        time.sleep(1.0 * (attempt + 1))
    return None


def search_video_ids(keyword, key_pool, max_results=VIDEOS_PER_KEYWORD):
    params = {
        "part": "id",
        "q": keyword,
        "type": "video",
        "maxResults": max_results,
        "relevanceLanguage": "vi",
        "regionCode": "VN",
    }
    resp = _get_with_retry(YOUTUBE_SEARCH_URL, params, key_pool)
    if resp is None:
        return []
    return [item["id"]["videoId"] for item in resp.json().get("items", [])]


MIN_COMMENT_COUNT = 5


def filter_videos_with_comments(video_ids, key_pool):
    if not video_ids:
        return []
    ok_ids = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        params = {"part": "statistics", "id": ",".join(batch)}
        resp = _get_with_retry(YOUTUBE_VIDEOS_URL, params, key_pool)
        if resp is None:
            continue
        for item in resp.json().get("items", []):
            stats = item.get("statistics", {})
            comment_count = int(stats.get("commentCount", 0))
            if comment_count >= MIN_COMMENT_COUNT:
                ok_ids.append(item["id"])
    return ok_ids


def fetch_comments(video_id, key_pool, max_pages=COMMENT_PAGES_PER_VIDEO):
    records = []
    page_token = None
    for _ in range(max_pages):
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": COMMENTS_PER_PAGE,
            "textFormat": "plainText",
            "order": "relevance",
        }
        if page_token:
            params["pageToken"] = page_token
        resp = _get_with_retry(YOUTUBE_COMMENTS_URL, params, key_pool)
        if resp is None:
            break
        data = resp.json()
        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            records.append({
                "text": top["textDisplay"],
                "source": f"youtube:{video_id}",
                "location": None,
                "crawl_time": top["publishedAt"],
            })
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)
    return records


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"seen_video_ids": [], "done_keywords": []}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_rows(rows, out_path=OUT_PATH):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    with open(out_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def transform_records(records, hint_region=None):
    raw_rows = []
    labeled_rows = []
    for r in records:
        cleaned = clean_text(r.get("text"))
        cleaned = expand_abbreviations(cleaned)
        if not is_valid_sentence(cleaned):
            continue
        raw_rows.append({
            "text": cleaned,
            "source": r.get("source", "unknown"),
        })
        label = assign_region(cleaned, location=r.get("location"))
        if label["label_source"] != "lexicon":
            continue
        best_score = label["match_scores"][label["region"]]
        if best_score < MIN_LEXICON_SCORE:
            continue
        labeled_rows.append({
            "dialect": cleaned,
            "standard": "",
            "region": label["region"],
            "sentiment": "",
            "source": r.get("source", "unknown"),
            "source_split": "",
            "dialect_features": json.dumps(label["match_scores"], ensure_ascii=False),
        })
    return raw_rows, labeled_rows


def run_etl_pipeline():
    if not YOUTUBE_API_KEYS:
        print("[EXTRACT] Chưa set YOUTUBE_API_KEY hoặc YOUTUBE_API_KEYS, bỏ qua crawl.")
        return []

    key_pool = KeyPool(YOUTUBE_API_KEYS)
    state = load_state()
    seen_video_ids = set(state["seen_video_ids"])
    done_keywords = set(state["done_keywords"])

    total_rows = 0
    try:
        for region, keywords in REGION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in done_keywords:
                    continue
                video_ids = search_video_ids(keyword, key_pool)
                new_ids = [v for v in video_ids if v not in seen_video_ids]
                ok_ids = filter_videos_with_comments(new_ids, key_pool)
                print(f"[EXTRACT] '{keyword}': {len(video_ids)} video, {len(new_ids)} video mới, "
                      f"{len(ok_ids)} video có bình luận")

                for video_id in ok_ids:
                    comments = fetch_comments(video_id, key_pool)
                    raw_rows, labeled_rows = transform_records(comments)
                    append_rows(raw_rows, RAW_OUT_PATH)
                    append_rows(labeled_rows, OUT_PATH)
                    total_rows += len(labeled_rows)

                seen_video_ids.update(new_ids)

                done_keywords.add(keyword)
                state["seen_video_ids"] = list(seen_video_ids)
                state["done_keywords"] = list(done_keywords)
                save_state(state)
    except QuotaExhausted as e:
        print(f"[EXTRACT] Dừng phiên crawl: {e}. Chạy lại script này vào phiên/ngày sau sẽ tiếp tục từ đây.")

    print(f"[LOAD] Tổng số dòng mới ghi thêm trong phiên này: {total_rows} -> {OUT_PATH}")
    return total_rows


if __name__ == "__main__":
    run_etl_pipeline()
