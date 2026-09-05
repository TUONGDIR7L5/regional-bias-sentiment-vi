import csv
import random
import re
from pathlib import Path

random.seed(42)

ROOT_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT_DIR / "data" / "raw" / "vidia2std"
OUT_DIR = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TOTAL = {"train": 40000, "dev": 5000, "test": 5000}
REGIONS = ["northern", "central", "southern"]


def region_targets(total):
    base, rem = divmod(total, 3)
    return {region: base + (1 if i < rem else 0) for i, region in enumerate(REGIONS)}


def read_csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows[0], rows[1:]


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[\"'\u2018\u2019\u201c\u201d]", "", s)
    s = re.sub(r"[.,!?;:…]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_and_clean_original():
    headers, raws = {}, {}
    for split in ("train", "dev", "test"):
        h, r = read_csv_rows(INPUT_DIR / f"{split}.csv")
        headers[split], raws[split] = h, r

    dia_idx = {s: headers[s].index("dialect") for s in raws}
    std_idx = {s: headers[s].index("standard") for s in raws}

    def drop_empty(rows, di, si):
        kept, n_empty = [], 0
        for row in rows:
            if not row[di].strip() or not row[si].strip():
                n_empty += 1
                continue
            kept.append(row)
        return kept, n_empty

    def drop_internal_dup(rows, di, si):
        seen_dia, seen_std, kept, n_dup = set(), set(), [], 0
        for row in rows:
            n_dia, n_std = norm(row[di]), norm(row[si])
            if n_dia in seen_dia or n_std in seen_std:
                n_dup += 1
                continue
            seen_dia.add(n_dia)
            seen_std.add(n_std)
            kept.append(row)
        return kept, n_dup

    cleaned, report = {}, {}
    for split in ("train", "dev", "test"):
        rows, n_empty = drop_empty(raws[split], dia_idx[split], std_idx[split])
        rows, n_dup = drop_internal_dup(rows, dia_idx[split], std_idx[split])
        cleaned[split] = rows
        report[split] = {"rong": n_empty, "trung_noi_bo": n_dup, "trung_cheo": 0}

    def drop_cross(rows, di, si, protected_dia, protected_std):
        before = len(rows)
        rows = [r for r in rows if norm(r[di]) not in protected_dia and norm(r[si]) not in protected_std]
        return rows, before - len(rows)

    test_protected_dia = {norm(r[dia_idx["test"]]) for r in cleaned["test"]}
    test_protected_std = {norm(r[std_idx["test"]]) for r in cleaned["test"]}
    cleaned["dev"], report["dev"]["trung_cheo"] = drop_cross(
        cleaned["dev"], dia_idx["dev"], std_idx["dev"], test_protected_dia, test_protected_std
    )

    dev_test_protected_dia = test_protected_dia | {norm(r[dia_idx["dev"]]) for r in cleaned["dev"]}
    dev_test_protected_std = test_protected_std | {norm(r[std_idx["dev"]]) for r in cleaned["dev"]}
    cleaned["train"], report["train"]["trung_cheo"] = drop_cross(
        cleaned["train"], dia_idx["train"], std_idx["train"], dev_test_protected_dia, dev_test_protected_std
    )

    print("=== Báo cáo dọn dữ liệu gốc ===")
    for split in ("train", "dev", "test"):
        rep = report[split]
        total_removed = rep["rong"] + rep["trung_noi_bo"] + rep["trung_cheo"]
        print(
            f"{split}: gốc {len(raws[split])} dòng -> giữ {len(cleaned[split])} dòng "
            f"(loại {total_removed}: rỗng={rep['rong']}, trùng nội bộ={rep['trung_noi_bo']}, "
            f"trùng chéo với tập khác={rep['trung_cheo']})"
        )

    return headers, dia_idx, std_idx, cleaned


SUBJECTS = [
    "Tôi", "Mình", "Em", "Tớ", "Bọn tôi", "Nhà tôi", "Cả nhà tôi", "Anh",
    "Chị", "Bố tôi", "Mẹ tôi", "Ông tôi", "Bà tôi", "Chúng tôi", "Tụi mình",
    "Xóm mình", "Lớp mình", "Công ty mình", "Đám bạn tôi", "Chồng tôi",
    "Vợ tôi", "Con bé nhà tôi", "Thằng cu nhà tôi", "Sếp tôi", "Đồng nghiệp tôi",
]

_TOPICS_A = [
    "thời tiết hôm nay", "món phở này", "bữa cơm tối nay", "con đường này",
    "cái áo mới mua", "bộ phim vừa xem", "buổi họp sáng nay", "kỳ thi vừa rồi",
    "chuyến du lịch vừa rồi", "trận bóng đêm qua", "quán cà phê đầu ngõ",
    "phiên chợ Tết năm nay", "cái điện thoại mới", "công việc dạo này",
    "sức khỏe dạo này", "buổi tiệc tối qua", "cuốn sách đang đọc",
    "bài hát mới ra", "căn nhà mới thuê", "chiếc xe mới mua",
    "kỳ nghỉ lễ vừa rồi", "buổi biểu diễn tối qua", "món chè này",
    "ly trà sữa này", "buổi phỏng vấn sáng nay", "tình hình giao thông giờ này",
    "dịch vụ ở quán này", "thái độ của nhân viên đó", "kết quả trận đấu",
    "chất lượng wifi ở đây", "giá cả ở chợ này", "không khí trong nhà",
    "tiến độ công trình", "buổi họp phụ huynh", "chuyến bay sáng nay",
    "món quà sinh nhật", "bộ phim hoạt hình mới", "trò chơi mới cài",
    "lớp học thêm này", "buổi tập gym sáng nay", "cái laptop mới mua",
    "chuyến xe buýt sáng nay", "buổi liên hoan cuối năm", "món bún riêu này",
    "cái quạt mới mua", "buổi hẹn cà phê chiều nay", "kỳ lương tháng này",
    "buổi tổng vệ sinh cuối tuần", "chuyến về quê lần này", "buổi thi vấn đáp",
]

_TOPICS_B = [
    "quán phở đầu ngõ", "chiếc xe máy mới mua", "bộ phim vừa xem tối qua", "món bún bò",
    "cái áo khoác mới", "buổi họp lớp cuối năm", "chuyến du lịch Đà Lạt", "trận bóng đá tối qua",
    "cửa hàng tạp hóa đầu hẻm", "con đường trước nhà", "khu chợ đầu mối",
    "quán cà phê góc phố", "lớp học thêm buổi tối", "bệnh viện gần nhà", "công viên mới xây",
    "khu chung cư đang ở", "tiệm bánh mì đầu phố", "siêu thị mini gần trường", "nhà hàng hải sản",
    "sân bay hôm đi công tác", "ga tàu lúc sáng sớm", "bến xe khách", "trường cấp ba ngày xưa",
    "thầy giáo dạy Toán", "cô giáo chủ nhiệm", "ông chủ quán", "anh đồng nghiệp mới",
    "cô hàng xóm cạnh nhà", "con mèo nhà tôi", "con chó nhỏ mới nuôi", "khu vườn sau nhà",
    "mạng wifi nhà mình", "dịch vụ giao hàng hôm nay",
    "ứng dụng đặt đồ ăn", "giá xăng tuần này", "giá gạo ngoài chợ", "lương tháng này",
    "buổi phỏng vấn xin việc", "hợp đồng công việc mới",
    "dự án đang làm dở", "anh sếp mới về", "phòng trọ đang thuê", "căn hộ chung cư mới mua",
    "khách sạn ở biển", "bãi biển hôm cả nhà đi chơi", "ngôi chùa đầu làng",
    "lễ hội của làng", "đám cưới đứa bạn thân", "sinh nhật của con", "buổi biểu diễn ca nhạc",
    "kênh Youtube hay xem",
    "cái group trên Facebook", "cơn mưa chiều nay",
    "cái tủ lạnh mới mua", "cái máy giặt cũ", "cái quạt điện trong phòng", "chiếc ví da mới",
    "đôi giày mới mua", "mái tóc vừa cắt", "bữa cơm tối qua",
    "cái điều hòa mới lắp", "buổi liên hoan cơ quan",
    "quán trà sữa mới mở", "ngày Tết năm nay", "vụ mùa năm nay",
    "cái ao cá sau vườn", "đàn gà mới nuôi", "kết quả học tập của con",
    "chiếc tivi mới mua", "cái bếp gas nhà mình", "buổi hẹn hò cuối tuần", "cái công viên nước",
]

TOPICS = list(dict.fromkeys(_TOPICS_A + _TOPICS_B))

ALL_ADJ = list(dict.fromkeys([
    "tuyệt vời", "rất đẹp", "ngon quá", "vui hết sức", "thích ơi là thích",
    "xuất sắc", "hài lòng lắm", "tốt hơn mong đợi nhiều", "đáng yêu",
    "dễ chịu vô cùng", "hạnh phúc lắm", "ưng ý quá", "tự hào lắm",
    "thoải mái hẳn", "nhẹ nhõm cả người", "chất lượng ngoài mong đợi",
    "trên cả tuyệt vời", "đáng đồng tiền bát gạo", "vừa ý ghê",
    "ngon", "xịn", "hay", "đẹp", "đã", "chu đáo", "tận tình", "đỉnh", "mê luôn", "ấn tượng",
    "tệ quá", "chán ơi là chán", "kinh khủng", "thất vọng tràn trề",
    "khó chịu vô cùng", "dở tệ", "mệt mỏi quá", "buồn ơi là buồn",
    "tức ơi là tức", "đáng thất vọng", "phiền phức quá", "lo lắng quá",
    "sợ thật sự", "kém quá", "bực mình ghê", "chẳng ra làm sao cả",
    "tồi tệ hết chỗ nói", "nản quá đi mất",
    "dở", "tệ", "chán", "xấu", "tệ hại", "phí tiền", "không đáng",
    "chậm trễ", "kém chất lượng", "lừa đảo", "cẩu thả", "mất thời gian",
    "bình thường thôi", "cũng tạm ổn", "không có gì đặc biệt",
    "vẫn như mọi khi", "tàm tạm", "ổn, không phàn nàn gì",
    "trung bình thôi", "cũng được, không tệ không hay",
    "không tốt cũng không xấu", "chưa có gì để bàn",
    "tạm được", "ổn", "như mọi lần", "vừa đủ dùng",
    "ở mức chấp nhận được", "không có gì nổi bật",
]))

TEMPLATES = [
    "{subj} thấy {topic} {adj}.",
    "{topic} hôm nay {adj}, {subj_lower} không ngờ tới.",
    "Nói thật, {topic} {adj} lắm.",
    "{subj} vừa trải nghiệm {topic}, phải công nhận là {adj}.",
    "Về {topic} thì {adj}, khỏi bàn cãi.",
    "Theo {subj_lower} thấy thì {topic} {adj}.",
    "{subj} vừa xong {topic}, cảm giác {adj}.",
    "Ai cũng bảo {topic} {adj}, {subj_lower} thấy đúng thật.",
    "{topic} khiến {subj_lower} cảm thấy {adj}.",
    "Thử qua {topic} rồi mới thấy {adj}.",
    "{subj} chấm {topic} này {adj}.",
    "Kể ra thì {topic} cũng {adj}.",
    "Đánh giá của {subj_lower} về {topic}: {adj}.",
    "{subj} không nghĩ {topic} lại {adj} đến thế.",
    "Phải nói {topic} lần này {adj}.",
    "Mọi người bảo {topic} {adj}.",
    "Cả nhà đều bảo {topic} {adj}.",
]


def build_standard(subj, topic, adj, template):
    s = template.format(subj=subj, subj_lower=subj[0].lower() + subj[1:], topic=topic, adj=adj)
    return s[0].upper() + s[1:]


NORTHERN_PHRASE_MAP = {
    "trả lời": "giả nhời", "ăn cơm chưa": "cơm nước gì chưa",
    "biết rồi": "biết dồi", "thế à": "thế cơ à",
    "chúng tôi": "bọn tớ", "không có": "có đâu", "bây giờ": "giờ",
}
NORTHERN_WORD_MAP = {
    "nhỉ": "nhở", "rất": "cực", "lắm": "vãi", "ngon": "đỉnh",
    "thích": "mê", "không": "chẳng", "nhiều": "vãi cả",
    "được": "đc", "rồi": "dồi", "tôi": "tớ", "mẹ": "u",
    "bố": "thầy", "bạn": "cậu", "giàu": "giầu", "vẫn": "vưỡn",
    "bớt": "vợi", "nhé": "nhế",
}
NORTHERN_PARTICLES = ["nhở", "cơ", "đấy", "cơ mà", "còn gì", "ấy chứ", "thật đấy"]

CENTRAL_PHRASE_MAP = {
    "như thế": "như rứa", "ở đâu": "ở mô", "đi đâu": "đi mô",
    "làm gì": "mần chi", "cái gì": "cái chi", "chúng tôi": "bọn choa",
    "không có": "mô có", "không có gì": "chi mô",
}
CENTRAL_WORD_MAP = {
    "tôi": "tui", "không": "nỏ", "gì": "chi", "đâu": "mô",
    "này": "ni", "kia": "tê", "đó": "nớ", "vậy": "rứa", "sao": "răng",
    "thế": "rứa", "nhé": "nghe", "nhỉ": "hỉ", "mẹ": "mạ", "bố": "bọ",
    "làm": "mần", "rất": "khiếp", "à": "hầy", "được": "đặng",
    "bà": "mệ", "bạn": "mi", "anh": "eng", "lắm": "trời luôn", "nữa": "nựa",
}
CENTRAL_PARTICLES = ["hè", "tề", "nờ", "rứa đó", "mà nói rứa", "hí"]

SOUTHERN_PHRASE_MAP = {
    "như thế": "như dzậy", "bây giờ": "giờ", "hôm nay": "bữa nay",
    "hôm nọ": "bữa đó", "một chút": "chút xíu",
    "chúng tôi": "tụi tui", "ông ấy": "ổng", "bà ấy": "bả",
    "cô ấy": "nhỏ đó", "vậy à": "dữ hôn", "không có": "hông có",
}
SOUTHERN_WORD_MAP = {
    "tôi": "tui", "không": "hổng", "này": "nè", "thế": "dzậy",
    "vậy": "dzậy", "lắm": "dữ lắm", "nhé": "nghen", "đấy": "đó",
    "kia": "đó", "mẹ": "má", "bố": "ba", "cha": "ba",
    "rất": "quá trời", "quá": "quá trời", "rồi": "rùi",
    "à": "hả", "vâng": "dạ", "nhỉ": "héng", "cực kỳ": "xỉu luôn",
    "thật": "thiệt",
}
SOUTHERN_PARTICLES = ["nghen", "à nha", "dữ lắm", "hen", "đó nha"]

REGION_CONF = {
    "northern": (NORTHERN_PHRASE_MAP, NORTHERN_WORD_MAP, NORTHERN_PARTICLES),
    "central": (CENTRAL_PHRASE_MAP, CENTRAL_WORD_MAP, CENTRAL_PARTICLES),
    "southern": (SOUTHERN_PHRASE_MAP, SOUTHERN_WORD_MAP, SOUTHERN_PARTICLES),
}


def apply_dialect_map(sentence, phrase_map, word_map):
    s = sentence
    for k in sorted(phrase_map, key=len, reverse=True):
        pattern = re.compile(re.escape(k), re.IGNORECASE)

        def repl(m, v=phrase_map[k]):
            orig = m.group(0)
            return v[0].upper() + v[1:] if orig[0].isupper() else v

        s = pattern.sub(repl, s)

    def word_repl(m):
        w = m.group(0)
        lw = w.lower()
        if lw in word_map:
            v = word_map[lw]
            return v[0].upper() + v[1:] if w[0].isupper() else v
        return w

    return re.sub(r"\w+", word_repl, s, flags=re.UNICODE)


def add_particle(sentence, particles, prob=0.45):
    if random.random() > prob or not sentence.endswith((".", "!", "?")):
        return sentence
    return sentence[:-1] + " " + random.choice(particles) + sentence[-1]


def dialectize(sentence, region):
    phrase_map, word_map, particles = REGION_CONF[region]
    s = apply_dialect_map(sentence, phrase_map, word_map)
    s = add_particle(s, particles)
    if s == sentence:
        s = add_particle(sentence, particles, prob=1.0)
    return s


def build_combo_pool():
    pool = [
        (template, topic, subj, adj)
        for template in TEMPLATES
        for topic in TOPICS
        for subj in SUBJECTS
        for adj in ALL_ADJ
    ]
    random.shuffle(pool)
    return pool


def generate_region_rows(region, count, combo_pool, pool_idx, seen_dialect, seen_standard, header, dia_i, std_i, region_i):
    rows = []
    i = pool_idx
    while len(rows) < count and i < len(combo_pool):
        template, topic, subj, adj = combo_pool[i]
        i += 1
        standard = build_standard(subj, topic, adj, template)
        n_std = norm(standard)
        if n_std in seen_standard:
            continue
        dialect = dialectize(standard, region)
        n_dia = norm(dialect)
        if n_dia in seen_dialect or n_dia == n_std:
            continue
        seen_standard.add(n_std)
        seen_dialect.add(n_dia)
        row = [""] * len(header)
        row[dia_i], row[std_i], row[region_i] = dialect, standard, region
        rows.append(row)
    return rows, i


def write_final(split, header, original_rows, synthetic_rows):
    out_path = OUT_DIR / f"{split}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(original_rows)
        writer.writerows(synthetic_rows)
    print(f"[{split}] đã ghi {len(original_rows)} dòng gốc + {len(synthetic_rows)} dòng sinh thêm = {len(original_rows) + len(synthetic_rows)} dòng -> {out_path}")


def region_count(rows, region_idx):
    from collections import Counter
    return Counter(r[region_idx] for r in rows)


def main():
    headers, dia_idx, std_idx, cleaned = load_and_clean_original()
    region_idx = {s: headers[s].index("region") for s in cleaned}

    seen_standard, seen_dialect = set(), set()
    for split, rows in cleaned.items():
        for r in rows:
            seen_standard.add(norm(r[std_idx[split]]))
            seen_dialect.add(norm(r[dia_idx[split]]))

    combo_pool = build_combo_pool()
    pool_idx = 0

    print("\n=== Sinh dữ liệu thêm theo vùng miền ===")
    for split in ("train", "dev", "test"):
        real_counts = region_count(cleaned[split], region_idx[split])
        targets = region_targets(TARGET_TOTAL[split])
        synthetic_rows = []
        for region in REGIONS:
            need = max(0, targets[region] - real_counts.get(region, 0))
            new_rows, pool_idx = generate_region_rows(
                region, need, combo_pool, pool_idx, seen_dialect, seen_standard,
                headers[split], dia_idx[split], std_idx[split], region_idx[split],
            )
            synthetic_rows += new_rows
            print(f"  [{split}][{region}] có sẵn {real_counts.get(region, 0)}, mục tiêu {targets[region]}, sinh thêm {len(new_rows)}")
        write_final(split, headers[split], cleaned[split], synthetic_rows)


if __name__ == "__main__":
    main()
