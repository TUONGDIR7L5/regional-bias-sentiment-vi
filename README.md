# Nghiên cứu phương pháp Domain Adaptation nhằm giảm thiểu thiên lệch vùng miền trong phân tích cảm xúc Tiếng Việt

Khóa luận tốt nghiệp — hướng đề tài: Các bài toán phân lớp (phân tích cảm xúc, phát hiện tin giả, nhận diện văn bản/lời nói độc hại).

## Mục tiêu

Xây dựng một bộ dữ liệu cảm xúc tiếng Việt được cân bằng và gán nhãn theo 3 vùng miền (Bắc – Trung – Nam), sau đó nghiên cứu và cải tiến các kỹ thuật Domain Adaptation (Adversarial DA qua Gradient Reversal Layer, External Knowledge Injection) nhằm giảm chênh lệch độ chính xác (Accuracy Gap) giữa các vùng miền khi phân loại cảm xúc.

## Nội dung thực hiện

1. **Xây dựng dữ liệu** — thu thập/sinh dữ liệu mới từ bộ gốc [ViDia2Std](.), tự động sinh thêm dữ liệu và crawl bổ sung từ báo chí + hội nhóm địa phương trên mạng xã hội; gán nhãn cảm xúc + vùng miền, đo Inter-Annotator Agreement (Cohen's Kappa).
2. **Baseline** — huấn luyện và so sánh 4 model BERT-based: PhoBERT, ViSoBERT, XLM-R, mBERT.
3. **Ablation study** — 3 cấu hình cải tiến độc lập trên model baseline tốt nhất:
   - **B1**: Adversarial Domain Adaptation (GRL)
   - **B2**: External Knowledge Injection (từ điển phương ngữ)
   - **B3**: kết hợp cả hai
4. **Cải tiến phương pháp** — GRL với trọng số động (dynamic weighting) + phân tích Explainable AI (xAI) trên attention weights.

## Cấu trúc thư mục

```
configs/        cấu hình dữ liệu, model, huấn luyện
data/           dữ liệu (raw → interim → processed), annotation
src/            toàn bộ code: thu thập dữ liệu, tiền xử lý, model, train, đánh giá, xAI
notebooks/      notebook khám phá/thử nghiệm, đánh số theo trình tự chạy
dashboard/      BI Dashboard trực quan hóa bias theo vùng miền
reports/        báo cáo (so sánh dataset, Model Fairness Report, xAI report)
results/        kết quả số liệu (metrics, log) theo từng nhóm model
tests/          unit test
```

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quy trình chạy dữ liệu (chi tiết)

Cần set biến môi trường trước khi chạy: `YOUTUBE_API_KEY` (crawl) và `GEMINI_API_KEY` (sinh dữ liệu tự động, dùng Gemini free tier).

```bash
# 1. Lấy raw data gốc từ Hugging Face (Biu3010/ViDia2Std)
python src/data_collection/fetch_vidia2std.py

# 2. Crawl thêm dữ liệu từ YouTube, gán vùng miền bằng bộ lọc từ vựng
#    Có thể set YOUTUBE_API_KEYS="key1,key2,key3" để xoay nhiều key free khi hết quota.
#    Script tự lưu tiến độ (data/raw/crawled/social_media/_state.json) và ghi thêm dần vào
#    crawled.csv, nên có thể chạy lại nhiều lần/nhiều ngày mà không mất dữ liệu đã thu được.
python src/data_collection/crawlers/social_media_crawler.py

# 3. Sinh dữ liệu tự động bằng Gemini API (few-shot theo phong cách ViDia2Std)
python src/data_collection/synthetic_generation/generate_from_vidia2std.py

# 4. Gộp 3 nguồn, lọc rỗng/quá ngắn, khử trùng lặp (raw vidia2std làm chuẩn ưu tiên)
python src/preprocessing/merge_and_dedup.py

# 5. Tạo batch gán nhãn cảm xúc: annotator_1 gán toàn bộ dòng còn thiếu nhãn,
#    annotator_2 gán một mẫu ~15% (chia đều theo vùng miền) để đo Cohen's Kappa
python src/annotation/prepare_labeling_batches.py
#    -> điền cột sentiment (positive/negative/neutral) trong
#       data/annotations/annotator_1/to_label.csv và annotator_2/to_label.csv

# 6. Gộp nhãn 2 người, tính Cohen's Kappa, xuất data/interim/merged_dedup_labeled.csv
python src/annotation/agreement_kappa.py

# 7. Trích đặc trưng phương ngữ + xây từ điển vùng miền
python src/dictionary/build_regional_dictionary.py
python src/preprocessing/dialect_feature_extractor.py

# 8. Chia train/dev/test 8:1:1, stratified theo vùng miền + cảm xúc
python src/preprocessing/stratified_split.py

# 9. So sánh định lượng với các bộ dữ liệu công khai
python src/preprocessing/compare_datasets.py
```

Có thể chạy tuần tự toàn bộ (trừ bước gán nhãn thủ công) bằng:

```bash
python run_pipeline.py
python run_pipeline.py --only merge_and_dedup
```

Script tự dừng ở bước `finalize_labels` nếu `annotator_1/to_label.csv` chưa được điền nhãn xong.

## Huấn luyện mô hình (sau khi có dữ liệu)

```bash
python src/training/train_baseline.py --model phobert
python src/training/train_ablation.py --variant b1
python src/training/train_improved.py
python src/evaluation/fairness_eval.py
```

## Nhóm thực hiện

Thesis_8688 — Kim Xuyến, Tường Vy

## Timeline

06/08/2026 – 23/01/2027, dự kiến bảo vệ (phản biện) đầu tháng 3/2027.
