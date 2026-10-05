# Phần 3: Phân tích và mô hình hóa
md("""
## 3. PHÂN TÍCH VÀ MÔ HÌNH HÓA

### 3.1. Thước đo đánh giá
Với 3 lớp và ~90% là lớp Tích cực, Accuracy rất dễ gây hiểu nhầm. Nhóm báo cáo đủ 5 thước đo theo yêu cầu đề bài, tính **trung bình macro**
(mỗi lớp có trọng số bằng nhau, nên lớp hiếm không bị lớp đông "nuốt"):
- **F1 macro** — thước đo chính để chọn mô hình (cân bằng Precision và Recall, công bằng giữa 3 lớp).
- **Precision macro, Recall macro, Accuracy.**
- **ROC-AUC macro (one-vs-rest)** — khả năng xếp hạng, không phụ thuộc ngưỡng; tính từ xác suất hoặc `decision_function` (với LinearSVC).
- **F1 lớp Trung tính** — theo dõi riêng vì đây là lớp khó nhất và hiếm nhất.

### 3.2. Baseline và so sánh các mô hình (trên Validation)
- **Baseline 1 — Dummy:** luôn đoán lớp đa số.
- **Baseline 2 — Logistic Regression** tham số mặc định, không xử lý mất cân bằng.
- **3 thuật toán theo đề:** MultinomialNB, Logistic Regression, LinearSVC — mỗi thuật toán chạy **hai phiên bản**: mặc định và phiên bản xử lý mất cân bằng
  (`class_weight='balanced'` cho LogReg/LinearSVC; `fit_prior=False` cho MultinomialNB, tức là dùng xác suất tiên nghiệm đều thay vì theo tần suất lớp).
""")

code(r"""
CLASSES = [0, 1, 2]


def get_scores(model, X):
    return model.predict_proba(X) if hasattr(model, 'predict_proba') else model.decision_function(X)


def macro_auc(y_true, scores):
    Y = label_binarize(y_true, classes=CLASSES)
    return float(np.mean([roc_auc_score(Y[:, k], scores[:, k]) for k in CLASSES]))


def evaluate(model, X, y_true):
    pred = model.predict(X)
    return {'Accuracy': accuracy_score(y_true, pred),
            'Precision': precision_score(y_true, pred, average='macro', zero_division=0),
            'Recall': recall_score(y_true, pred, average='macro', zero_division=0),
            'F1': f1_score(y_true, pred, average='macro', zero_division=0),
            'ROC-AUC': macro_auc(y_true, get_scores(model, X)),
            'F1 Trung tính': f1_score(y_true, pred, labels=[1], average='macro', zero_division=0)}


METRIC_FMT = {c: '{:.3f}' for c in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'F1 Trung tính']}

ALGORITHMS = {
    'MultinomialNB': {'default': lambda: MultinomialNB(), 'balanced': lambda: MultinomialNB(fit_prior=False)},
    'LogisticRegression': {'default': lambda: LogisticRegression(max_iter=2000),
                           'balanced': lambda: LogisticRegression(max_iter=2000, class_weight='balanced')},
    'LinearSVC': {'default': lambda: LinearSVC(random_state=RANDOM_STATE),
                  'balanced': lambda: LinearSVC(class_weight='balanced', random_state=RANDOM_STATE)},
}

models = {'Dummy (lớp đa số)': make_pipeline(DummyClassifier(strategy='most_frequent'))}
for name, versions in ALGORITHMS.items():
    models[name] = make_pipeline(versions['default']())
    models[f'{name} + balanced'] = make_pipeline(versions['balanced']())

val_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    val_rows.append({'Mô hình': name, **evaluate(model, X_val, y_val)})
val_results = pd.DataFrame(val_rows).set_index('Mô hình')
display(val_results.style.format(METRIC_FMT).highlight_max(color='#c6efce', axis=0))
""")
