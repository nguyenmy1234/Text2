# 🛒 Bài 2 — Dữ liệu văn bản (Text): Phân loại cảm xúc đánh giá sản phẩm Amazon

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange.svg)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green.svg)

**Bài toán:** phân loại đa lớp (3 lớp) cảm xúc của đánh giá sản phẩm Amazon (tiếng Anh): **Tiêu cực / Trung tính / Tích cực**.
Nhãn suy ra từ điểm sao: 1–2 sao → Tiêu cực, 3 sao → Trung tính, 4–5 sao → Tích cực.

**Dữ liệu:** `amazon_reviews.csv` — 4,915 đánh giá × 12 cột ([Kaggle: tarkkaanko/amazon](https://www.kaggle.com/datasets/tarkkaanko/amazon)); sau khi xóa 1 dòng thiếu `reviewText` và 2 dòng trùng còn **4,912** đánh giá (90.5% Tích cực / 6.6% Tiêu cực / 2.9% Trung tính).
`external_test.jsonl` — 5,000 đánh giá ngoài (1,000 mẫu mỗi nhãn 0–4) **chỉ dùng để kiểm tra**, không dùng huấn luyện.

## 📁 Cấu trúc thư mục
```
.
├── data/text/
│   ├── amazon_reviews.csv            # dữ liệu chính (4,915 dòng)
│   └── external_test.jsonl           # tập kiểm tra ngoài (5,000 dòng)
├── scripts/
│   ├── prepare_amazon_dataset.py     # kiểm tra dữ liệu / tạo external_test.jsonl
│   ├── generate_text_notebook.py     # ghép nb_parts -> notebook (và chạy nếu thêm --execute)
│   └── nb_parts/                     # nội dung notebook, chia theo phần
│       ├── part1_eda.py              # Phần 0–1: môi trường, mô tả dữ liệu, EDA
│       ├── part2_preparation.py      # Phần 2: làm sạch, chia Train/Val/Test, Pipeline
│       ├── part3_modeling.py         # Phần 3: baseline, so sánh, GridSearchCV, chọn mô hình
│       ├── part4_summary.py          # Phần 4: Test, phân tích lỗi, tập ngoài, Transformer, predict_sample
│       └── notes.py                  # các đoạn nhận xét (kèm số liệu thật) chèn sau từng ô
├── notebooks/
│   └── Amazon_Reviews.ipynb         # notebook đã chạy sẵn (có output)
├── result/
│   └── Amazon_Reviews.html         # bản HTML của notebook đã chạy
├── docs/
│   └── index.html                   # trang giới thiệu (GitHub Pages)
├── requirements.txt
└── README.md
```

## 🚀 Cách chạy

```bash
pip install -r requirements.txt

# 1) Kiểm tra dữ liệu
python scripts/prepare_amazon_dataset.py

# 2) Sinh notebook + chạy toàn bộ + lưu kết quả (mất vài phút, GridSearchCV 5-fold)
python scripts/generate_text_notebook.py --execute
```

Hoặc chỉ sinh notebook rồi mở bằng Jupyter / Google Colab và chạy **Run All**:
```bash
python scripts/generate_text_notebook.py
jupyter notebook notebooks/Amazon_Reviews.ipynb
```

- Notebook tìm dữ liệu ở `data/text`, `../data/text`, `/content`; nếu không thấy sẽ tự tải từ GitHub (đặt `GITHUB_RAW_BASE` ở ô 1.1 cho đúng repo của nhóm).
- Mục 4.6 (RoBERTa zero-shot) cần `pip install torch transformers` và internet; nếu thiếu, ô đó tự bỏ qua, các phần khác vẫn chạy bình thường.

## 🔬 Quy trình
1. **EDA** — thiếu/trùng, phân bố nhãn, độ dài, tương quan Spearman, outlier IQR.
2. **Chuẩn bị** — chia 70/15/15 phân tầng **trước** khi fit; `TextCleaner` (giữ từ phủ định) → TF-IDF (5,000 đặc trưng, 1–2-gram) trong `Pipeline` để tránh rò rỉ dữ liệu.
3. **Mô hình** — Dummy, MultinomialNB, Logistic Regression, LinearSVC (mặc định vs xử lý mất cân bằng), GridSearchCV 5-fold theo F1 macro, thử thêm meta-features.
4. **Tổng kết** — đánh giá Test một lần (Accuracy, Precision, Recall, F1, ROC-AUC macro), phân tích lỗi, kiểm tra tập ngoài, hàm `predict_sample`.

## 📊 Kết quả chính (Test, 737 mẫu; mô hình cuối: Logistic Regression `C=3`, `class_weight='balanced'`)
| Chỉ số | Giá trị |
|---|---|
| Accuracy | 0.919 |
| F1 macro | 0.593 (Dummy 0.317; LogReg mặc định 0.458) |
| ROC-AUC macro | 0.884 |
| Bắt được Tiêu cực / Trung tính | 37/48 và 3/22 (baseline: 13/48 và 0/22) |
| Tập ngoài (5,000 mẫu) F1 macro | 0.389 → khó tổng quát hóa sang ngành hàng khác |

> Số liệu lấy từ lần chạy với `random_state=42` (Python 3.12, pandas 3.0.2, scikit-learn 1.8.0); phiên bản thư viện khác có thể lệch nhẹ ở chữ số cuối.

## 🌐 GitHub Pages
Trang giới thiệu nằm ở `docs/index.html` (kèm `result/Amazon_Reviews.html`, nút "Xem notebook" trỏ tới file này).

## ⚠️ Cần điền trước khi nộp
- Tên nhóm và danh sách thành viên ở đầu `scripts/nb_parts/part1_eda.py` và ở `docs/index.html` (tìm `<Tên nhóm>`).
- `GITHUB_RAW_BASE` ở ô 1.1 nếu repo không phải `perckgg/NTLT`.
