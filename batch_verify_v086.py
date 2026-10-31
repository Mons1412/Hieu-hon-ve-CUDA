#!/usr/bin/env python3
"""One-screen human relevance review against complete ChromaDB chunks.

Usage: python batch_verify_v086.py MM01 [MM02 ...]
       python batch_verify_v086.py              # all pending rows

Never fills final_relevance automatically; saves each human choice atomically.
"""

import csv
import os
import sys
from pathlib import Path

import chromadb
import pymupdf
from pdf_rag import DB_DIR

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "eval_results/relevance_verified_v084.csv"
PDF_PATH = ROOT / "docs/pdf/CUDA_C_Best_Practices_Guide.pdf"
COLLECTIONS = {"BASELINE": "cuda_pdf_v03", "SECTION_AWARE": "cuda_pdf_section_v073"}
VALID_SCORES = {"0", "1", "2"}


def normalize(value):
    return " ".join(value.split())


def load_rows():
    if not CSV_PATH.is_file():
        raise SystemExit(f"FAIL: Missing data file: {CSV_PATH}")
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"id", "question", "collection", "rank", "page", "evidence",
                    "suggested_relevance", "final_relevance", "verification"}
        if not required.issubset(fields):
            raise SystemExit(f"FAIL: Missing CSV columns: {sorted(required - set(fields))}")
        rows = list(reader)
    keys = [(r["id"], r["collection"], r["rank"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("FAIL: Duplicate (id, collection, rank) keys")
    return fields, rows


def save_rows(fields, rows):
    temp = CSV_PATH.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, CSV_PATH)


def count_done(rows):
    return sum(r["final_relevance"] in VALID_SCORES for r in rows)


def show_page(page):
    if not PDF_PATH.is_file():
        print(f"FAIL: Missing PDF: {PDF_PATH}")
        return False
    with pymupdf.open(PDF_PATH) as pdf:
        if not 1 <= page <= len(pdf):
            print(f"FAIL: Invalid PDF page {page} (PDF has {len(pdf)} pages)")
            return False
        print(f"\n{'=' * 20} FULL PDF PAGE {page} {'=' * 20}")
        print(pdf[page - 1].get_text())
    return True


def main():
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Usage: python batch_verify_v086.py [SM01 MM01 ...]")
        return
    fields, rows = load_rows()
    chosen_ids = set(sys.argv[1:])
    unknown = chosen_ids - {r["id"] for r in rows}
    if unknown:
        raise SystemExit(f"FAIL: Unknown question IDs: {sorted(unknown)}")

    db = chromadb.PersistentClient(path=str(DB_DIR))
    collections = {}  # Retrieve each collection once, only if needed
    print(f"START: {count_done(rows)}/{len(rows)} final labels saved")
    print("0=irrelevant | 1=partial | 2=direct | p=full PDF | s=skip | q=quit")

    for row in rows:
        if chosen_ids and row["id"] not in chosen_ids:
            continue
        if row["final_relevance"] in VALID_SCORES:
            continue

        group = row["collection"]
        if group not in COLLECTIONS:
            print(f"SKIP: Unknown collection group: {group}")
            continue
        if group not in collections:
            collections[group] = db.get_collection(COLLECTIONS[group])
        page = int(row["page"])
        result = collections[group].get(
            where={"page": page}, include=["documents"]
        )
        needle = normalize(row["evidence"].strip())
        matches = list(dict.fromkeys(
            doc for doc in result["documents"]
            if normalize(doc).startswith(needle)
        ))

        print("\n" + "=" * 66)
        print(f"{row['id']} | {group} | Rank {row['rank']} | PDF page {page}")
        print("QUESTION:", row["question"])
        print("ORIGINAL:", row.get("original_relevance", ""))
        print("AI SUGGESTION:", row["suggested_relevance"])
        print("AI REASON:", row.get("rationale_vi", ""))

        if len(matches) != 1:
            print(f"\nMATCH WARNING: Found {len(matches)} chunks, expected 1.")
            print("This record was NOT scored. Inspect the match logic before review.")
            continue
        print("\nFULL CHUNK:\n", matches[0], sep="")
        inspected_pdf = False

        while True:
            action = input("\nScore [0/1/2] | p=PDF | s=skip | q=quit: ").strip().lower()
            if action == "p":
                inspected_pdf = show_page(page) or inspected_pdf
            elif action in VALID_SCORES:
                row["final_relevance"] = action
                row["verification"] = (
                    "HUMAN_REVIEWED_WITH_PDF" if inspected_pdf
                    else "HUMAN_REVIEWED_FULL_CHUNK"
                )
                save_rows(fields, rows)
                print(f"SAVED: {row['id']} {group} Rank {row['rank']} = {action}")
                print(f"PROGRESS: {count_done(rows)}/{len(rows)}")
                break
            elif action == "s":
                print("SKIPPED: label unchanged")
                break
            elif action == "q":
                print(f"STOPPED: {count_done(rows)}/{len(rows)} labels saved")
                return
            else:
                print("INVALID INPUT. Use 0, 1, 2, p, s, or q.")

    print(f"DONE: {count_done(rows)}/{len(rows)} labels saved")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Previously saved labels are preserved.")
