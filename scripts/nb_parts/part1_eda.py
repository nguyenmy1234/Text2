# Phần 0 + 1: Giới thiệu, môi trường, mô tả dữ liệu & EDA (theo khung EDA của bài 01_Tabular)
md("""
# BÀI TẬP LỚN — NỀN TẢNG LẬP TRÌNH CHO PHÂN TÍCH VÀ TRỰC QUAN DỮ LIỆU
**Học kỳ 261 – Năm học 2026–2027 · Giảng viên: Lê Thành Sách**

## BÀI 2 — DỮ LIỆU VĂN BẢN (TEXT): PHÂN LOẠI CẢM XÚC ĐÁNH GIÁ SẢN PHẨM AMAZON
- **Nhóm:** `<Tên nhóm>` — **Thành viên:** `<Họ tên – MSSV – vai trò>`

**Bài toán:** phân loại đa lớp (3 lớp) — dự đoán cảm xúc của một đánh giá sản phẩm Amazon (tiếng Anh) là
**Tiêu cực / Trung tính / Tích cực**, để bộ phận chăm sóc khách hàng nhận diện sớm các đánh giá cần phản hồi.

**Câu hỏi chính của bài:**
1. Những từ và đặc điểm nào phân biệt đánh giá tiêu cực, trung tính và tích cực?
2. Dữ liệu **mất cân bằng nặng** (~90% tích cực) ảnh hưởng thế nào, và **xử lý mất cân bằng giúp bắt thêm được bao nhiêu đánh giá tiêu cực / trung tính**?

**Quy trình:** EDA → làm sạch văn bản và tiền xử lý trong Pipeline (chống rò rỉ dữ liệu) → so sánh baseline với 3 thuật toán
(MultinomialNB, Logistic Regression, LinearSVC) → tinh chỉnh siêu tham số bằng GridSearchCV → đánh giá một lần trên tập Test →
phân tích lỗi, kiểm tra trên tập ngoài và hàm suy luận `predict_sample`.
""")

md("## 0. CÀI ĐẶT MÔI TRƯỜNG")

code(r"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
sns.set_theme(style='whitegrid')
RANDOM_STATE = 42
CLASS_NAMES = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}
CLASS_COLORS = {0: '#DD5145', 1: '#E0B341', 2: '#55A868'}
print('Môi trường đã sẵn sàng!')
""")

md("""
## 1. MÔ TẢ TẬP DỮ LIỆU VÀ KHÁM PHÁ (EDA)

**Nguồn:** bộ đánh giá sản phẩm Amazon (chủ yếu là thẻ nhớ SD/microSD, đánh giá từ 01/2012 đến 12/2014; xem bằng chứng ở 1.9), file `amazon_reviews.csv`
gồm **4,915 đánh giá × 12 cột**. Mỗi dòng có điểm sao `overall` (1–5) do chính người mua chấm và nội dung `reviewText`.
**Link nguồn:** https://www.kaggle.com/datasets/tarkkaanko/amazon.

**Cách gán nhãn cảm xúc** (theo gợi ý của đề: Tích cực / Trung tính / Tiêu cực):

| Điểm sao `overall` | Nhãn | Mã |
|---|---|---|
| 1 – 2 sao | Tiêu cực (Negative) | 0 |
| 3 sao | Trung tính (Neutral) | 1 |
| 4 – 5 sao | Tích cực (Positive) | 2 |

> ⚠️ **Lưu ý minh bạch:** đây là bộ dữ liệu công khai có sẵn, **nhóm không tự crawl**. Nhãn cảm xúc suy ra từ điểm sao, nên có nhiễu
> (người dùng có thể viết nội dung tiêu cực nhưng vẫn chấm 3–4 sao). Phần phân tích lỗi (4.4) sẽ kiểm tra điều này.
> Ngoài ra nhóm dùng thêm file `external_test.jsonl` (5,000 đánh giá, nhãn 0–4) **chỉ làm tập kiểm tra ngoài** ở mục 4.5, không dùng để huấn luyện.

### 1.1. Tải dữ liệu
""")

code(r"""
GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/perckgg/NTLT/main/02_Text/data/text'
DATA_FILE = 'amazon_reviews.csv'
EXTERNAL_FILE = 'external_test.jsonl'
LOCAL_DIRS = ['data/text', '../data/text', '/content/data/text', '/content']


def locate(filename):
    for d in LOCAL_DIRS:
        p = Path(d) / filename
        if p.exists():
            return str(p)
    return f'{GITHUB_RAW_BASE}/{filename}'


def read_csv_safe(filename):
    path = locate(filename)
    try:
        return pd.read_csv(path), path
    except Exception as err:
        raise FileNotFoundError(
            f'Không đọc được {filename} ({path}). Hãy upload file vào /content hoặc push thư mục 02_Text/data/text '
            f'lên GitHub và chỉnh GITHUB_RAW_BASE. Lỗi gốc: {err}')


def read_jsonl_safe(filename):
    path = locate(filename)
    try:
        return pd.read_json(path, lines=True), path
    except Exception as err:
        raise FileNotFoundError(f'Không đọc được {filename} ({path}). Lỗi gốc: {err}')


raw, data_path = read_csv_safe(DATA_FILE)
print(f'Nguồn dữ liệu: {data_path}')
print(f'Kích thước: {raw.shape[0]:,} dòng × {raw.shape[1]} cột')
display(raw.head(3))
""")

md("""
### 1.2. Ý nghĩa các cột và kiểu dữ liệu
""")

code(r"""
COLUMN_INFO = {
    'Unnamed: 0': 'Chỉ số dòng của file gốc (định danh, không mang thông tin)',
    'reviewerName': 'Tên người đánh giá (định danh, cardinality rất cao)',
    'overall': 'Điểm sao 1–5 do người mua chấm → dùng để gán nhãn cảm xúc',
    'reviewText': 'Nội dung đánh giá (tiếng Anh) → đặc trưng văn bản chính',
    'reviewTime': 'Ngày đăng đánh giá',
    'day_diff': 'Số ngày từ lúc đăng đánh giá đến ngày thu thập dữ liệu',
    'helpful_yes': 'Số người bình chọn "hữu ích"',
    'helpful_no': 'Số người bình chọn "không hữu ích"',
    'total_vote': 'Tổng số bình chọn (= helpful_yes + helpful_no)',
    'score_pos_neg_diff': 'helpful_yes − helpful_no (suy ra từ cột bình chọn)',
    'score_average_rating': 'helpful_yes / total_vote (suy ra từ cột bình chọn)',
    'wilson_lower_bound': 'Cận dưới Wilson của tỉ lệ hữu ích (suy ra từ cột bình chọn)',
}
info = pd.DataFrame({'Kiểu dữ liệu': raw.dtypes.astype(str), 'Số giá trị khác nhau': raw.nunique(),
                     'Ý nghĩa': pd.Series(COLUMN_INFO)})
display(info)
""")

md("""
### 1.3. Tổng quan nhanh (KPI)
Các chỉ số tổng quan giống bảng điều khiển phân tích đánh giá thương mại điện tử: số đánh giá, điểm sao trung bình,
độ dài đánh giá trung bình, số ngày kể từ khi đăng và số phiếu "hữu ích".
""")

code(r"""
kpi = pd.Series({
    'Tổng số đánh giá': len(raw),
    'Điểm sao trung bình': raw['overall'].mean(),
    'Độ dài trung bình (số từ)': raw['reviewText'].dropna().str.split().str.len().mean(),
    'Số ngày kể từ khi đăng (trung bình)': raw['day_diff'].mean(),
    'Tổng phiếu "hữu ích"': raw['helpful_yes'].sum(),
    '% đánh giá có người bình chọn': (raw['total_vote'] > 0).mean() * 100,
})
display(kpi.to_frame('Giá trị').style.format('{:,.2f}'))
""")

md("""
### 1.4. Giá trị thiếu (Missing values)
""")

code(r"""
miss = pd.DataFrame({'Số ô thiếu': raw.isna().sum(), 'Tỉ lệ (%)': raw.isna().mean() * 100})
miss['Mức độ'] = pd.cut(miss['Tỉ lệ (%)'], [-0.001, 0, 5, 20, 100], labels=['Không thiếu', 'Thấp (<5%)', 'Trung bình (5–20%)', 'Cao (>20%)'])
display(miss)
print(f'Số dòng có ít nhất 1 ô thiếu: {raw.isna().any(axis=1).sum()} ({raw.isna().any(axis=1).mean():.2%})')
print('Dòng thiếu reviewText:')
display(raw[raw['reviewText'].isna()])
print(f"Số đánh giá có nội dung trùng lặp: {raw['reviewText'].dropna().duplicated().sum()}")
""")

md(note("MISSING"))

md("""
### 1.5. Phân bố biến mục tiêu
""")

code(r"""
def rating_to_label(r):
    return 0 if r <= 2 else (1 if r == 3 else 2)


df_eda = raw.dropna(subset=['reviewText']).copy()
df_eda['label'] = df_eda['overall'].map(rating_to_label)
df_eda['word_count'] = df_eda['reviewText'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
star = df_eda['overall'].value_counts().sort_index()
axes[0].bar(star.index.astype(int), star.values, color=['#DD5145', '#DD5145', '#E0B341', '#55A868', '#55A868'])
for x, v in zip(star.index.astype(int), star.values):
    axes[0].text(x, v + 40, f'{v:,}\n({v / len(df_eda):.1%})', ha='center', fontsize=9)
axes[0].set(title='Phân bố điểm sao (overall)', xlabel='Số sao', ylabel='Số đánh giá', ylim=(0, star.max() * 1.18))

lab = df_eda['label'].value_counts().sort_index()
axes[1].bar([CLASS_NAMES[i] for i in lab.index], lab.values, color=[CLASS_COLORS[i] for i in lab.index])
for x, v in enumerate(lab.values):
    axes[1].text(x, v + 40, f'{v:,}\n({v / len(df_eda):.1%})', ha='center', fontsize=9)
axes[1].set(title='Phân bố nhãn cảm xúc (3 lớp)', ylabel='Số đánh giá', ylim=(0, lab.max() * 1.18))
plt.tight_layout(); plt.show()

ratio = lab.max() / lab.min()
print(f'Tỉ lệ lớp đa số / lớp thiểu số = {ratio:.1f} : 1')
print(f'Mô hình luôn đoán "Tích cực" đã đạt Accuracy = {lab.max() / lab.sum():.3f} nhưng F1 trung bình các lớp chỉ = '
      f'{f1_score(df_eda["label"], np.full(len(df_eda), 2), average="macro"):.3f}')
""")

md(note("TARGET"))
