import json
import re
from pathlib import Path

import pymupdf

PDF = Path("docs/pdf/CUDA_C_Best_Practices_Guide.pdf")
OUTPUT = Path("eval_results/page53_sections.json")

with pymupdf.open(PDF) as pdf:
    page_number = 53
    lines = pdf[page_number - 1].get_text("text").splitlines()

# Phát hiện điểm bắt đầu mục 10.2.3.3
boundary = next(
    (i for i, line in enumerate(lines)
     if re.match(r"^10\.2\.3\.3\s+", line.strip())),
    None
)

if boundary is None:
    raise SystemExit(
        "FAIL: Không tìm được heading 10.2.3.3"
    )

# Phần đầu trang thuộc mục trước đó
previous = lines[:boundary]

# Phần sau heading thuộc mục mới
current = lines[boundary:]

# Bỏ footer của trang PDF
footer = next(
    (i for i, line in enumerate(current)
     if line.startswith("10.2. Device Memory Spaces")),
    len(current)
)

current = current[:footer]

sections = [
    {
        "source": PDF.name,
        "page": 53,
        "section": "10.2.3.2",
        "topic": "Matrix Multiplication C=AB",
        "text": "\n".join(previous).strip()
    },
    {
        "source": PDF.name,
        "page": 53,
        "section": "10.2.3.3",
        "topic": "Matrix Multiplication C=AAT",
        "text": "\n".join(current).strip()
    }
]

# Kiểm tra ranh giới bằng nội dung thực tế
assert "Table 2" in sections[0]["text"]
assert "195.5 GB/s" in sections[0]["text"]

assert "simpleMultiply" in sections[1]["text"]
assert "12.8 GB/s" in sections[1]["text"]

OUTPUT.parent.mkdir(exist_ok=True)

OUTPUT.write_text(
    json.dumps(sections, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

for item in sections:
    print("\nSECTION:", item["section"])
    print("TOPIC:", item["topic"])
    print("PAGE:", item["page"])
    print("CHARACTERS:", len(item["text"]))
    print("PREVIEW:", item["text"][:250])

print("\nSECTION SPLIT: PASS")
print("OUTPUT:", OUTPUT)
