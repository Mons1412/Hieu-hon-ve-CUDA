import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from api import search_evidence
from rag_answer_v043 import generate_answer

CASES = [
    {
        "id": "BANK_CONFLICTS",
        "question": "Shared Memory có tự động loại bỏ Bank Conflicts không?",
        "expected_pages": [49, 54],
        "expected_status": "ANSWER",
        "review_hint": "Không tự loại bỏ; phần cứng xử lý conflict."
    },
    {
        "id": "MATRIX_MULTIPLICATION",
        "question": "Shared Memory giúp tối ưu Matrix Multiplication như thế nào?",
        "expected_pages": [50, 52, 53],
        "expected_status": "ANSWER",
        "review_hint": "Tái sử dụng tile, giảm truy cập Global Memory dư thừa."
    },
    {
        "id": "CONTROL_FLOW",
        "question": "What is Control Flow?",
        "expected_pages": [],
        "expected_status": "REVIEW",
        "review_hint": "Kiểm tra nguồn trước khi quyết định ANSWER hay ABSTAIN."
    },
    {
        "id": "OUT_OF_SCOPE",
        "question": "Thủ đô của Pháp là gì?",
        "expected_pages": [],
        "expected_status": "ABSTAIN",
        "review_hint": "Tài liệu CUDA không cung cấp căn cứ về địa lý."
    }
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    output = Path("eval_results")
    output.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for case in CASES:
        print("\n==========", case["id"], "==========", flush=True)
        result = dict(case)

        try:
            sources = search_evidence(case["question"], 4)
            pages = [s["page"] for s in sources]

            hit = (
                bool(set(pages) & set(case["expected_pages"]))
                if case["expected_pages"] else None
            )

            if args.retrieval_only:
                status = "SKIPPED"
                answer = None
                reason = "Retrieval-only mode"
            else:
                status, answer, reason = generate_answer(
                    case["question"], sources
                )

            result.update(
                pages=pages,
                retrieval_hit=hit,
                status=status,
                answer=answer,
                reason=reason,
                sources=[
                    {
                        "page": s["page"],
                        "source": s["source"],
                        "text": s["text"]
                    }
                    for s in sources
                ]
            )

        except Exception as exc:
            result.update(
                pages=[],
                retrieval_hit=False if case["expected_pages"] else None,
                status="ERROR",
                answer=None,
                reason=str(exc),
                sources=[]
            )

        results.append(result)

        print("PAGES:", result["pages"])
        print("STATUS:", result["status"])
        print("ANSWER:", result["answer"])
        print("REASON:", result["reason"])

    json_path = output / f"rag_eval_{timestamp}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    csv_path = output / f"manual_review_{timestamp}.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "id", "status", "expected_status",
            "retrieval_hit", "pages", "answer",
            "review_hint", "human_verdict", "review_notes"
        ])

        for r in results:
            writer.writerow([
                r["id"],
                r["status"],
                r["expected_status"],
                r["retrieval_hit"],
                r["pages"],
                r["answer"],
                r["review_hint"],
                "",
                ""
            ])

    retrieval_cases = [
        r for r in results if r["expected_pages"]
    ]

    status_cases = [
        r for r in results
        if r["expected_status"] != "REVIEW"
        and r["status"] != "SKIPPED"
    ]

    print("\n========== SUMMARY ==========")

    print(
        "RETRIEVAL HIT@4:",
        sum(r["retrieval_hit"] for r in retrieval_cases),
        "/", len(retrieval_cases)
    )

    if status_cases:
        print(
            "STATUS MATCH:",
            sum(
                r["status"] == r["expected_status"]
                for r in status_cases
            ),
            "/", len(status_cases)
        )

    print("FAITHFULNESS: NEEDS MANUAL REVIEW")
    print("RESULT:", json_path)
    print("REVIEW FILE:", csv_path)


if __name__ == "__main__":
    main()
