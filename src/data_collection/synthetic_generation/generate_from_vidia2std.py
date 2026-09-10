import csv
import json
import os
import random
import re
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from google import genai
from google.genai import types

from src.preprocessing.cleaning import clean_text

random.seed(42)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ROOT_DIR = Path(__file__).resolve().parents[3]
VIDIA2STD_DIR = ROOT_DIR / "data" / "raw" / "vidia2std"
OUT_PATH = ROOT_DIR / "data" / "raw" / "synthetic" / "synthetic.csv"

REGIONS = ["northern", "central", "southern"]
SENTIMENTS = ["positive", "negative", "neutral"]
TARGET_TOTAL = 15000
BATCH_SIZE = 20
FEWSHOT_PER_REGION = 6
SLEEP_BETWEEN_CALLS = 4
MAX_RETRIES = 3
COLUMNS = ["dialect", "standard", "region", "sentiment", "source", "source_split"]


def region_sentiment_targets(total):
    n_cells = len(REGIONS) * len(SENTIMENTS)
    base = total // n_cells
    targets = {}
    for region in REGIONS:
        for sentiment in SENTIMENTS:
            targets[(region, sentiment)] = base
    return targets


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[\"'\u2018\u2019\u201c\u201d]", "", s)
    s = re.sub(r"[.,!?;:…]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_fewshot_examples():
    path = VIDIA2STD_DIR / "train.csv"
    examples = {region: [] for region in REGIONS}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    random.shuffle(rows)
    for row in rows:
        region = row.get("region")
        if region in examples and len(examples[region]) < FEWSHOT_PER_REGION:
            examples[region].append((row["dialect"], row["standard"]))
        if all(len(v) >= FEWSHOT_PER_REGION for v in examples.values()):
            break
    return examples


def build_prompt(region, sentiment, count, fewshot):
    fewshot_text = "\n".join(f"- {dia} || {std}" for dia, std in fewshot)
    return f"""Bạn là chuyên gia ngôn ngữ học tiếng Việt vùng miền.
Hãy viết {count} câu bình luận mạng xã hội bằng phương ngữ vùng {region} (northern/central/southern) của Việt Nam,
mang sắc thái cảm xúc "{sentiment}" (positive/negative/neutral), nội dung đời thường, đa dạng chủ đề, KHÔNG lặp cấu trúc câu.
Ví dụ phong cách phương ngữ vùng {region} (câu phương ngữ || câu chuẩn tương ứng):
{fewshot_text}

Trả lời DUY NHẤT bằng JSON array, mỗi phần tử dạng:
{{"dialect": "...", "standard": "..."}}
Không thêm giải thích, không thêm markdown."""


def call_gemini(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.9,
                    max_output_tokens=4096,
                    top_p=0.95,
                ),
            )
            return response.text
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "503" in msg or "UNAVAILABLE" in msg:
                time.sleep(5 * (attempt + 1))
                continue
            print(f"[GENERATE] Loi API: {msg[:300]}")
            return None
    return None


def parse_response(text):
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and "dialect" in d and "standard" in d]


def load_existing_progress(out_path=OUT_PATH):
    seen_dialect, seen_standard = set(), set()
    done_counts = {}
    if not out_path.exists():
        return seen_dialect, seen_standard, done_counts
    with open(out_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen_dialect.add(norm(row["dialect"]))
            seen_standard.add(norm(row["standard"]))
            key = (row["region"], row["sentiment"])
            done_counts[key] = done_counts.get(key, 0) + 1
    return seen_dialect, seen_standard, done_counts


def append_rows(rows, out_path=OUT_PATH):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    with open(out_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def generate_all():
    if not GEMINI_API_KEY:
        print("[GENERATE] Chưa set biến môi trường GEMINI_API_KEY, bỏ qua sinh dữ liệu.")
        return 0

    client = genai.Client(api_key=GEMINI_API_KEY)
    fewshot = load_fewshot_examples()
    targets = region_sentiment_targets(TARGET_TOTAL)
    seen_dialect, seen_standard, done_counts = load_existing_progress()
    total_new = 0

    for (region, sentiment), target_count in targets.items():
        collected = done_counts.get((region, sentiment), 0)
        if collected >= target_count:
            print(f"[GENERATE] {region}/{sentiment}: đã đủ {collected}/{target_count}, bỏ qua")
            continue
        while collected < target_count:
            remaining = target_count - collected
            batch = min(BATCH_SIZE, remaining)
            prompt = build_prompt(region, sentiment, batch, fewshot[region])
            response = call_gemini(client, prompt)
            time.sleep(SLEEP_BETWEEN_CALLS)
            if response is None:
                print(f"[GENERATE] Dừng ở {region}/{sentiment}: {collected}/{target_count}, "
                      f"chạy lại script sẽ tiếp tục từ đây.")
                break
            items = parse_response(response)
            if not items:
                continue
            new_rows = []
            for item in items:
                dialect = clean_text(item.get("dialect"))
                standard = clean_text(item.get("standard"))
                if not dialect or not standard:
                    continue
                n_dia, n_std = norm(dialect), norm(standard)
                if n_dia in seen_dialect or n_std in seen_standard:
                    continue
                seen_dialect.add(n_dia)
                seen_standard.add(n_std)
                new_rows.append({
                    "dialect": dialect,
                    "standard": standard,
                    "region": region,
                    "sentiment": sentiment,
                    "source": "synthetic",
                    "source_split": "",
                })
                collected += 1
                if collected >= target_count:
                    break
            append_rows(new_rows)
            total_new += len(new_rows)
            print(f"[GENERATE] {region}/{sentiment}: {collected}/{target_count}")

    print(f"[LOAD] Tổng số dòng mới ghi thêm trong phiên này: {total_new} -> {OUT_PATH}")
    return total_new


def run_etl_pipeline():
    return generate_all()


if __name__ == "__main__":
    run_etl_pipeline()
