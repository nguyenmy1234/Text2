"""Ghép các phần trong scripts/nb_parts thành notebook notebooks/Amazon_Reviews.ipynb.

Mỗi file part*.py được thực thi với 3 hàm trợ giúp:
    md(text)   -> thêm một ô Markdown
    code(text) -> thêm một ô Code
    note(key)  -> trả về đoạn nhận xét (markdown) lấy từ notes.py

Cách dùng (chạy từ thư mục gốc):
    python scripts/generate_text_notebook.py            # chỉ sinh notebook
    python scripts/generate_text_notebook.py --execute  # sinh + chạy toàn bộ + lưu kết quả vào result/
"""
import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
PARTS_DIR = ROOT / "scripts" / "nb_parts"
PARTS = ["part1_eda.py", "part2_preparation.py", "part3_modeling.py", "part4_summary.py"]
NOTEBOOK = ROOT / "notebooks" / "Amazon_Reviews.ipynb"

sys.path.insert(0, str(PARTS_DIR))
from notes import note  # noqa: E402


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = nb["cells"]
    ns = {"md": lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n"))),
          "code": lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n"))),
          "note": note}
    for part in PARTS:
        path = PARTS_DIR / part
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}")
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def execute(nb):
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="chạy toàn bộ notebook và lưu kèm kết quả")
    args = ap.parse_args()
    nb = build_notebook()
    if args.execute:
        print("Đang chạy toàn bộ notebook (vài phút)...")
        nb = execute(nb)
        (ROOT / "result").mkdir(exist_ok=True)
        from nbconvert import HTMLExporter
        html, _ = HTMLExporter().from_notebook_node(nb)
        (ROOT / "result" / "Amazon_Reviews.html").write_text(html, encoding="utf-8")
    NOTEBOOK.parent.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOK)
    print(f"Đã ghi: {NOTEBOOK.relative_to(ROOT)} ({len(nb.cells)} ô)")


if __name__ == "__main__":
    main()
