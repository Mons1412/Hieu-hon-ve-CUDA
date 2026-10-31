import csv
from pathlib import Path

DATA = """
SM01|Shared Memory|Shared Memory trong CUDA là gì?
SM02|Shared Memory|Shared Memory khác Global Memory như thế nào?
SM03|Shared Memory|Tại sao Shared Memory có độ trễ thấp?
SM04|Shared Memory|Các thread trong block chia sẻ dữ liệu như thế nào?
SM05|Shared Memory|Shared Memory có thể hoạt động như cache không?
GM01|Global Memory|Global Memory trong CUDA là gì?
GM02|Global Memory|Coalesced Memory Access là gì?
GM03|Global Memory|Uncoalesced Access ảnh hưởng hiệu suất thế nào?
GM04|Global Memory|Làm thế nào giảm truy cập Global Memory dư thừa?
BC01|Bank Conflicts|Shared Memory có tự động loại bỏ Bank Conflicts không?
BC02|Bank Conflicts|Bank Conflicts xảy ra khi nào?
BC03|Bank Conflicts|Padding giúp tránh Bank Conflicts như thế nào?
MM01|Matrix Multiplication|Shared Memory giúp tối ưu Matrix Multiplication như thế nào?
MM02|Matrix Multiplication|Tiling trong Matrix Multiplication là gì?
MM03|Matrix Multiplication|Tại sao cần tái sử dụng dữ liệu giữa các thread?
MM04|Matrix Multiplication|Shared Memory giúp giảm đọc tile B như thế nào?
OOS01|Out-of-scope|Thủ đô của Pháp là gì?
OOS02|Out-of-scope|Cách nấu phở bò ngon nhất?
OOS03|Out-of-scope|Ai vô địch World Cup 2022?
OOS04|Out-of-scope|Thời tiết Hà Nội hôm nay thế nào?
""".strip()

suggestions = {
    "BC01": (
        "ANSWER", "49,54",
        "Không. Shared Memory vẫn có thể xảy ra bank conflicts."
    ),
    "MM01": (
        "ANSWER", "50,52,53",
        "Shared Memory giúp tái sử dụng tile và giảm đọc Global Memory."
    )
}

rows = []

for line in DATA.splitlines():
    case_id, topic, question = line.split("|", 2)

    status, pages, answer = suggestions.get(
        case_id, ("REVIEW", "", "")
    )

    if topic == "Out-of-scope":
        status = "ABSTAIN"

    rows.append({
        "id": case_id,
        "topic": topic,
        "question": question,
        "proposed_status": status,
        "proposed_pages": pages,
        "reference_answer": answer,
        "verified": "NO"
    })

assert len(rows) == 20

path = Path("golden_candidates_v061.csv")

with path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("GOLDEN CANDIDATES CREATED:", len(rows))
print("FILE:", path)
print("VERIFIED: 0/20 — Manual review required")
