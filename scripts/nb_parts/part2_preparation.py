# Phần 2: Chuẩn bị dữ liệu
md("""
## 2. CHUẨN BỊ DỮ LIỆU

Nguyên tắc: mọi bước **có học tham số từ dữ liệu** (từ vựng và IDF của TF-IDF, median để điền thiếu, trung bình/độ lệch chuẩn để scale)
đều nằm trong `Pipeline` và chỉ được `fit` trên tập **Train** → tránh rò rỉ dữ liệu (data leakage) sang Validation/Test.
Riêng bước làm sạch văn bản không học tham số từ dữ liệu nên được đóng gói thành một lớp `TextCleaner` dùng lại được (OOP).

### 2.1. Xử lý giá trị thiếu, trùng lặp, loại cột không dùng và gán nhãn
""")

code(r"""
n_raw = len(raw)
data = raw.copy()

# (1) Giá trị thiếu: không thể phân loại cảm xúc khi không có nội dung -> xóa dòng thiếu reviewText.
data = data.dropna(subset=['reviewText'])
n_after_na = len(data)

# (2) Trùng lặp: xóa đánh giá có nội dung giống hệt (giữ bản đầu) để không bị lặp giữa Train và Test.
data = data.drop_duplicates(subset='reviewText')
n_after_dup = len(data)

# (3) Loại cột không dùng: định danh, ngày đăng, và 3 cột suy ra từ cột bình chọn (trùng thông tin).
DROP_COLS = ['Unnamed: 0', 'reviewerName', 'reviewTime', 'score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound']
data = data.drop(columns=DROP_COLS)

# (4) Gán nhãn cảm xúc từ điểm sao rồi bỏ cột overall khỏi đặc trưng (tránh rò rỉ nhãn).
data['label'] = data['overall'].map(rating_to_label)
stars = data['overall'].astype(int)
data = data.drop(columns='overall').reset_index(drop=True)
stars = stars.reset_index(drop=True)
TARGET = 'label'

print(f'Ban đầu: {n_raw:,} dòng | sau khi xóa thiếu reviewText: {n_after_na:,} (-{n_raw - n_after_na}) | '
      f'sau khi xóa trùng: {n_after_dup:,} (-{n_after_na - n_after_dup})')
print(f'Cột còn lại: {list(data.columns)}')
print(f'Còn giá trị thiếu không: {int(data.isna().sum().sum())}')
display(data['label'].map(CLASS_NAMES).value_counts().to_frame('Số mẫu').assign(**{'Tỉ lệ': lambda d: d['Số mẫu'] / len(data)}))
""")

md(note("CLEAN"))

md("""
### 2.2. Chia tập Train / Validation / Test (70 / 15 / 15, phân tầng theo nhãn)
- **Train:** huấn luyện. **Validation:** so sánh mô hình, chọn cấu hình. **Test:** chỉ dùng **một lần** ở Phần 4.
- `stratify=y` giữ tỉ lệ ba lớp ở cả ba tập (quan trọng vì lớp Trung tính rất hiếm). Chia **trước** khi fit TF-IDF/scaler để tránh rò rỉ.
""")

code(r"""
BASE_COLS = ['reviewText', 'day_diff', 'helpful_yes', 'helpful_no', 'total_vote']
X, y = data[BASE_COLS], data[TARGET]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)

split_df = pd.DataFrame({name: {'Số mẫu': len(t), 'Tỉ lệ': len(t) / len(X), **{f'% {CLASS_NAMES[k]}': (t == k).mean() * 100 for k in range(3)},
                                **{f'Số {CLASS_NAMES[k]}': int((t == k).sum()) for k in range(3)}}
                        for name, t in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]}).T
display(split_df)
""")
