import csv
import shutil
import sys
from pathlib import Path

import chromadb
import pymupdf
from pdf_rag import DB_DIR, get_collection

if len(sys.argv) != 4:
    raise SystemExit(
        "Usage: python verify_label_v084.py SM01 BASELINE 1"
    )

case_id, group, rank = sys.argv[1:]
src = Path("eval_results/relevance_suggestions_v083.csv")
dst = Path("eval_results/relevance_verified_v084.csv")

if not dst.exists():
    shutil.copy2(src, dst)

with dst.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

matches = [
    r for r in rows
    if r["id"] == case_id
    and r["collection"] == group
    and r["rank"] == rank
]

if len(matches) != 1:
    raise SystemExit("FAIL: Record not uniquely identified")

row = matches[0]
page = int(row["page"])

db = chromadb.PersistentClient(path=str(DB_DIR))
col = (
    get_collection()
    if group == "BASELINE"
    else db.get_collection("cuda_pdf_section_v073")
)

data = col.get(
    where={"page": page},
    include=["documents"]
)

snippet = row["evidence"].strip()
chunks = [
    doc for doc in data["documents"]
    if " ".join(doc.split()).startswith(snippet)
]

if len(chunks) != 1:
    raise SystemExit("FAIL: Cannot match full chunk uniquely")

pdf_path = "docs/pdf/CUDA_C_Best_Practices_Guide.pdf"

with pymupdf.open(pdf_path) as pdf:
    pdf_text = pdf[page - 1].get_text()

print("\nQUESTION:", row["question"])
print("SOURCE:", group, "RANK:", rank, "PAGE:", page)
print("ORIGINAL:", row["original_relevance"])
print("AI SUGGESTION:", row["suggested_relevance"])
print("AI REASON:", row["rationale_vi"])

print("\n========== FULL CHUNK ==========")
print(chunks[0])

print("\n========== FULL PDF PAGE ==========")
print(pdf_text)

score = input("\nFinal score [0/1/2], s=skip: ").strip()

if score not in {"0", "1", "2"}:
    raise SystemExit("SKIPPED: No label changed")

row["final_relevance"] = score
row["verification"] = "HUMAN_REVIEWED_WITH_PDF"

tmp = dst.with_suffix(".tmp")

with tmp.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

tmp.replace(dst)

verified = sum(
    r["final_relevance"] in {"0", "1", "2"}
    for r in rows
)

print("\nLABEL SAVED:", case_id, group, rank)
print("FINAL SCORE:", score)
print(f"VERIFIED PROGRESS: {verified}/{len(rows)}")
