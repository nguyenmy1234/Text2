# Phần 4: Tổng kết và mở rộng
md("""
## 4. TỔNG KẾT VÀ MỞ RỘNG

### 4.1. Đánh giá trên tập Test độc lập
Mô hình, siêu tham số và cấu hình đặc trưng đều đã được chốt ở Phần 3 (dựa trên Validation); tập Test chỉ được dùng **một lần** ở đây.
Bảng dưới so sánh tất cả mô hình trên cùng tập Test với 5 thước đo của đề (Accuracy, Precision, Recall, F1, ROC-AUC; trung bình macro).
""")

code(r"""
test_models = {'Dummy (lớp đa số)': models['Dummy (lớp đa số)'], 'LogisticRegression (baseline)': models['LogisticRegression'], **candidates}
test_results = pd.DataFrame({k: evaluate(m, X_test, y_test) for k, m in test_models.items()}).T
display(test_results.style.format(METRIC_FMT).highlight_max(color='#c6efce', axis=0))

test_pred = final_model.predict(X_test)
test_scores = get_scores(final_model, X_test)
print(f'=== {best_name} — TẬP TEST ({len(y_test)} mẫu) ===')
print(classification_report(y_test, test_pred, target_names=[CLASS_NAMES[k] for k in CLASSES], digits=3, zero_division=0))
""")

md(note("TEST"))
