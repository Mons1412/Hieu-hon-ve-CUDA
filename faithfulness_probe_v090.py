#!/usr/bin/env python3
"""v0.9: Compare actual Qwen answers given fair-scope Top-4 retrieval.

Read-only for ChromaDB. Saves a JSON evidence record and CSV review form.
Faithfulness is deliberately NOT auto-graded.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

CASE_IDS = ("BC01", "MM01", "MM03", "OOS01")
COLLECTION_NAMES = ("BASELINE", "SECTION_AWARE")
FIELDS = (
    "id", "question", "collection", "expected_status", "actual_status",
    "status_match", "cited_ranks", "pages", "answer", "reason",
    "latency_seconds", "human_faithfulness", "human_notes",
)


def atomic_save(records: list[dict], stem: Path) -> None:
    """Persist after each run so a later interruption does not lose results."""
    json_path = stem.with_suffix(".json")
    json_tmp = stem.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    json_tmp.replace(json_path)

    csv_path = stem.with_suffix(".csv")
    csv_tmp = stem.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r["id"],
                "question": r["question"],
                "collection": r["collection"],
                "expected_status": r["expected_status"],
                "actual_status": r["actual_status"],
                "status_match": r["status_match"],
                "cited_ranks": ",".join(str(x) for x in r["cited_ranks"]),
                "pages": ",".join(str(x["page"]) for x in r["sources"]),
                "answer": r["answer"] or "",
                "reason": r["reason"],
                "latency_seconds": r["latency_seconds"],
                "human_faithfulness": "",
                "human_notes": "",
            })
    csv_tmp.replace(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA RAG Answer Faithfulness v0.9")
    parser.add_argument("--ids", nargs="+", choices=CASE_IDS, default=list(CASE_IDS))
    parser.add_argument(
        "--collections", nargs="+", choices=COLLECTION_NAMES,
        default=list(COLLECTION_NAMES)
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Inspect retrieved passages without invoking Qwen"
    )
    args = parser.parse_args()

    # Dependencies and DB access are postponed until after CLI validation.
    import chromadb
    from ollama import embed
    from pdf_rag import DB_DIR, EMBED_MODEL, get_collection
    from rag_answer_v043 import generate_answer

    cases_path = Path("golden_candidates_v061.csv")
    with cases_path.open(encoding="utf-8-sig", newline="") as handle:
        available = {r["id"]: r for r in csv.DictReader(handle)}
    for case_id in args.ids:
        if case_id not in available:
            raise SystemExit(f"Missing case in {cases_path}: {case_id}")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collections = {}
    if "BASELINE" in args.collections:
        collections["BASELINE"] = get_collection()
    if "SECTION_AWARE" in args.collections:
        collections["SECTION_AWARE"] = client.get_collection(
            "cuda_pdf_section_v073"
        )
    for name, collection in collections.items():
        count = collection.count()
        print(f"COLLECTION {name}: {count} chunks", flush=True)
        if count < 4:
            raise SystemExit(f"Collection {name} missing/empty; not indexing automatically")

    output_dir = Path("eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = output_dir / f"faithfulness_v090_{run_id}"
    records = []

    print("RETRIEVAL POLICY: single original query; page >= 7; cosine Top-4", flush=True)
    print("GENERATOR: rag_answer_v043.generate_answer (unchanged)", flush=True)
    print("FAITHFULNESS: manual review required", flush=True)

    for case_id in args.ids:
        question = available[case_id]["question"]
        expected = "ABSTAIN" if case_id.startswith("OOS") else "ANSWER"
        try:
            vector = list(embed(model=EMBED_MODEL, input=question).embeddings[0])
        except Exception as exc:
            print(f"EMBEDDING ERROR for {case_id}: {type(exc).__name__}: {exc}", flush=True)
            raise

        for name, collection in collections.items():
            print(f"\n===== {case_id} / {name} =====", flush=True)
            start = time.perf_counter()
            sources = []
            status, answer, reason = "ERROR", None, "not run"
            try:
                result = collection.query(
                    query_embeddings=[vector],
                    n_results=4,
                    where={"page": {"$gte": 7}},
                    include=["documents", "metadatas", "distances"],
                )
                docs = result["documents"][0]
                metas = result["metadatas"][0]
                dists = result["distances"][0]
                if len(docs) != 4:
                    raise RuntimeError(f"Expected exactly 4 results; got {len(docs)}")
                for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
                    sources.append({
                        "rank": rank,
                        "page": int(meta["page"]),
                        "section": str(meta.get("section", "")),
                        "source": str(meta.get("source", "")),
                        "distance": round(float(dist), 5),
                        "text": doc,  # FULL passage, not the 800-character CSV preview
                    })
                print("PAGES:", [s["page"] for s in sources], flush=True)
                if args.retrieval_only:
                    status, reason = "SKIPPED", "retrieval-only mode"
                else:
                    status, answer, reason = generate_answer(question, sources)
            except Exception as exc:
                status, answer = "ERROR", None
                reason = f"{type(exc).__name__}: {exc}"

            cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer or "")})
            record = {
                "id": case_id,
                "question": question,
                "collection": name,
                "expected_status": expected,
                "actual_status": status,
                "status_match": status == expected if status != "SKIPPED" else None,
                "cited_ranks": cited,
                "sources": sources,
                "answer": answer,
                "reason": reason,
                "latency_seconds": round(time.perf_counter() - start, 2),
                "retrieval_policy": "single_original_query_page_gte_7_top4",
                "generator": "rag_answer_v043.generate_answer",
                "human_faithfulness": None,
            }
            records.append(record)
            atomic_save(records, stem)
            print("STATUS:", status, "EXPECTED:", expected, flush=True)
            print("ANSWER:", answer, flush=True)
            print("REASON:", reason, flush=True)
            print("SAVED:", stem.with_suffix(".json"), flush=True)

    print("\n===== COMPLETED =====")
    print("CASES:", len(records))
    print("JSON:", stem.with_suffix(".json"))
    print("REVIEW CSV:", stem.with_suffix(".csv"))
    print("IMPORTANT: status/citation syntax does not prove factual faithfulness")


if __name__ == "__main__":
    main()
