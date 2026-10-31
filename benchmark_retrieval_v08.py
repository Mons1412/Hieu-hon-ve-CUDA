import csv
from pathlib import Path

import chromadb

from pdf_rag import DB_DIR, get_collection, retrieve

DATASET = Path("golden_candidates_v061.csv")
OUTPUT = Path("eval_results/retrieval_review_v08.md")

with DATASET.open(encoding="utf-8-sig") as f:
    cases = list(csv.DictReader(f))

client = chromadb.PersistentClient(path=str(DB_DIR))

old = get_collection()
new = client.get_collection("cuda_pdf_section_v073")

print("BASELINE CHUNKS:", old.count())
print("SECTION-AWARE CHUNKS:", new.count())

assert old.count() == 177
assert new.count() == 164

lines = [
    "# Retrieval Benchmark v0.8",
    "",
    "Baseline: cuda_pdf_v03",
    "",
    "Candidate: cuda_pdf_section_v073",
    "",
    "Evaluation cases: 20",
    "",
    "Results require manual relevance verification.",
    "",
]

def pages(results):
    return [
        r["metadata"]["page"]
        for r in results
    ]

for case in cases:
    case_id = case["id"]
    question = case["question"]

    old_results = retrieve(old, question, evidence_limit=4)
    new_results = retrieve(new, question, evidence_limit=4)

    print(f"\n{case_id}")
    print("OLD:", pages(old_results))
    print("NEW:", pages(new_results))

    lines.extend([
        f"## {case_id}",
        "",
        f"**Question:** {question}",
        "",
        f"**Old pages:** {pages(old_results)}",
        "",
        f"**New pages:** {pages(new_results)}",
        "",
    ])

    for name, results in [
        ("BASELINE", old_results),
        ("SECTION-AWARE", new_results),
    ]:
        lines.append(f"### {name}")
        lines.append("")

        for rank, item in enumerate(results, 1):
            meta = item.get("metadata") or {}

            page = meta.get("page", "?")
            section = meta.get("section", "N/A")

            text = (
                item.get("document")
                or item.get("text")
                or item.get("content")
                or ""
            )

            text = " ".join(str(text).split())

            lines.extend([
                f"**Rank {rank} — Page {page} — Section {section}**",
                "",
                text[:600] if text else "[No text in result]",
                "",
            ])

    lines.extend([
        "**Manual review:** PENDING",
        "",
        "---",
        "",
    ])

OUTPUT.parent.mkdir(exist_ok=True)

OUTPUT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\nBENCHMARK COMPLETED")
print("QUESTIONS:", len(cases))
print("REPORT:", OUTPUT)
print("RELEVANCE: NOT YET VERIFIED")
