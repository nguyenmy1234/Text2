"""Chuẩn bị và kiểm tra dữ liệu cho bài 02_Text (chạy từ thư mục gốc của repo).

1. Kiểm tra data/text/amazon_reviews.csv (4,915 dòng x 12 cột, đúng các cột cần dùng).
2. (Tùy chọn) Tạo data/text/external_test.jsonl: 5,000 đánh giá lấy phân tầng (1,000 mẫu mỗi nhãn 0-4)
   từ file jsonl lớn (cột: id, text, label, label_text), random_state=42.

Cách dùng:
    python scripts/prepare_amazon_dataset.py
    python scripts/prepare_amazon_dataset.py --source /duong/dan/train.jsonl   # tạo lại external_test.jsonl
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'text'
CSV = DATA_DIR / 'amazon_reviews.csv'
EXTERNAL = DATA_DIR / 'external_test.jsonl'
EXPECTED_COLS = ['Unnamed: 0', 'reviewerName', 'overall', 'reviewText', 'reviewTime', 'day_diff', 'helpful_yes',
                 'helpful_no', 'total_vote', 'score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound']
N_PER_CLASS = 1000


def check_csv():
    if not CSV.exists():
        raise SystemExit(f'Không tìm thấy {CSV}. Hãy đặt amazon_reviews.csv vào data/text/.')
    df = pd.read_csv(CSV)
    assert list(df.columns) == EXPECTED_COLS, f'Sai tên cột: {list(df.columns)}'
    print(f'[OK] {CSV.relative_to(ROOT)}: {df.shape[0]:,} dòng x {df.shape[1]} cột')
    print('     Phân bố điểm sao:', df['overall'].value_counts().sort_index().to_dict())
    print(f"     Thiếu reviewText: {int(df['reviewText'].isna().sum())} | trùng reviewText: {int(df['reviewText'].dropna().duplicated().sum())}")


def build_external(source):
    src = Path(source)
    if not src.exists():
        raise SystemExit(f'Không tìm thấy {src}')
    df = pd.read_json(src, lines=True)
    missing = {'id', 'text', 'label', 'label_text'} - set(df.columns)
    if missing:
        raise SystemExit(f'File nguồn thiếu cột: {missing}')
    small = (df.groupby('label', group_keys=False).sample(N_PER_CLASS, random_state=42)
               .sample(frac=1, random_state=42).reset_index(drop=True))
    small[['id', 'text', 'label', 'label_text']].to_json(EXTERNAL, orient='records', lines=True, force_ascii=False)
    print(f'[OK] Đã ghi {EXTERNAL.relative_to(ROOT)}: {len(small):,} dòng | {small["label"].value_counts().sort_index().to_dict()}')


def check_external():
    if not EXTERNAL.exists():
        print(f'[!] Chưa có {EXTERNAL.relative_to(ROOT)} (chạy lại với --source <file.jsonl> để tạo).')
        return
    ext = pd.read_json(EXTERNAL, lines=True)
    print(f'[OK] {EXTERNAL.relative_to(ROOT)}: {len(ext):,} dòng | nhãn: {ext["label"].value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', help='file jsonl lớn để lấy mẫu tạo external_test.jsonl')
    args = ap.parse_args()
    check_csv()
    if args.source:
        build_external(args.source)
    check_external()
