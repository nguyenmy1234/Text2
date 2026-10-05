# Nhận xét (markdown) chèn sau các ô code của notebook. Mỗi khóa được gọi bằng note("KHÓA") trong các part.
# Mọi con số dưới đây lấy từ lần chạy thực tế (random_state=42); chạy lại với phiên bản thư viện khác có thể lệch nhẹ ở chữ số thập phân cuối.
NOTES = {
    "MISSING": """
> **Nhận xét:**
> - Dữ liệu gần như đầy đủ: chỉ **2/4,915 dòng (0.04%)** có ô thiếu: 1 dòng thiếu `reviewText` và 1 dòng thiếu `reviewerName` (cột định danh, sẽ bị loại).
> - Dòng thiếu `reviewText` không thể phân loại cảm xúc nên sẽ bị **xóa** (không điền bù).
> - Có **2 đánh giá trùng nội dung** (`reviewText` giống hệt) → xóa bản lặp để cùng một câu không rơi vào cả Train lẫn Test.
""",
    "TARGET": """
> **Nhận xét:**
> - Điểm sao lệch hẳn về cao: **4–5 sao chiếm 4,448/4,914 (90.5%)**, 1–2 sao chỉ 324 (6.6%), 3 sao chỉ 142 (2.9%).
> - Tỉ lệ lớp đa số / thiểu số = **31.3 : 1**. Một mô hình "luôn đoán Tích cực" đạt Accuracy **0.905** nhưng F1 macro chỉ **0.317** → Accuracy không đủ để đánh giá; F1 macro được chọn làm thước đo chính.
> - Lớp **Trung tính chỉ có 142 mẫu** (khoảng 99 mẫu ở Train, 21–22 mẫu ở Validation/Test), nên mọi chỉ số của lớp này rất nhiễu.
""",
}


def note(key):
    return NOTES.get(key, f"> **Nhận xét ({key}):** _(chưa điền)_").strip("\n")
