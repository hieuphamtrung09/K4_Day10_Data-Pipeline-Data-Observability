from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_test_set(df: pd.DataFrame, output_path: str | Path) -> list[dict[str, Any]]:
    """Tạo bộ evaluation testset từ dataframe đã được làm sạch (cleaned dataframe).

    Args:
        df: DataFrame chứa dữ liệu sạch (cần có các cột doi/paper_id, title, summary, authors, published_date, subject).
        output_path: Đường dẫn file JSON để xuất kết quả.

    Returns:
        List các dictionary đại diện cho từng câu hỏi testset.
    """
    # 1. Kiểm tra số lượng document tối thiểu
    if df.empty:
        raise ValueError("DataFrame rỗng, không thể tạo testset.")

    # Đảm bảo cột ID chính xác (dùng doi hoặc paper_id)
    id_col = "doi" if "doi" in df.columns else "paper_id" if "paper_id" in df.columns else df.columns[0]

    test_samples: list[dict[str, Any]] = []
    sample_id = 1

    # 2. Duyệt qua từng paper đại diện để tạo các loại câu hỏi
    for _, row in df.iterrows():
        doc_id = str(row[id_col])
        title = str(row.get("title", "")).strip()
        summary = str(row.get("summary", "")).strip()
        authors = str(row.get("authors", "")).strip()
        published_date = str(row.get("published_date", "")).strip()
        subject = str(row.get("subject", "")).strip()

        # Bỏ qua nếu thiếu title hoặc summary tối thiểu
        if not title or not summary:
            continue

        # 3. Tạo các loại câu hỏi đa dạng (question_type)
        
        # Type 1: Summary / Nội dung chính
        test_samples.append({
            "id": f"eval_{sample_id:03d}",
            "question_type": "summary",
            "question": f"Tóm tắt nội dung chính của bài báo '{title}'?",
            "ground_truth": summary,
            "ground_truth_doc_ids": [doc_id],
        })
        sample_id += 1

        # Type 2: Authors / Tác giả
        if authors and authors.lower() != "nan":
            test_samples.append({
                "id": f"eval_{sample_id:03d}",
                "question_type": "authors",
                "question": f"Tác giả của bài báo '{title}' là ai?",
                "ground_truth": authors,
                "ground_truth_doc_ids": [doc_id],
            })
            sample_id += 1

        # Type 3: Date / Ngày xuất bản
        if published_date and published_date.lower() != "nan":
            test_samples.append({
                "id": f"eval_{sample_id:03d}",
                "question_type": "date",
                "question": f"Bài báo '{title}' được xuất bản vào thời gian nào?",
                "ground_truth": published_date,
                "ground_truth_doc_ids": [doc_id],
            })
            sample_id += 1

        # Type 4: Categories / Chủ đề
        if subject and subject.lower() != "nan":
            test_samples.append({
                "id": f"eval_{sample_id:03d}",
                "question_type": "categories",
                "question": f"Bài báo '{title}' thuộc chủ đề hoặc danh mục nào?",
                "ground_truth": subject,
                "ground_truth_doc_ids": [doc_id],
            })
            sample_id += 1

    # 4. Ghi file JSON vào output_path
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    return test_samples
