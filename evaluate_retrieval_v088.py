import csv
from collections import defaultdict
from pathlib import Path

path = Path("eval_results/relevance_working_v087.csv")

with path.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 160, "Expected 160 retrieval results"

groups = defaultdict(list)
sources = defaultdict(int)

for r in rows:
    assert r["working_relevance"] in {"0", "1", "2"}
    key = (r["id"], r["collection"])
    groups[key].append(r)
    sources[r["label_source"]] += 1

assert len(groups) == 40, "Expected 20 questions x 2 collections"

for key, items in groups.items():
    ranks = sorted(int(r["rank"]) for r in items)
    assert ranks == [1, 2, 3, 4], f"Invalid ranks: {key}"

def scores(items):
    items = sorted(items, key=lambda r: int(r["rank"]))
    relevant = [
        int(r["rank"])
        for r in items
        if r["working_relevance"] == "2"
    ]

    hit = int(bool(relevant))
    precision = len(relevant) / 4
    rr = 1 / relevant[0] if relevant else 0

    return hit, precision, rr

print("========== PRELIMINARY RETRIEVAL METRICS ==========")
print("TOTAL ROWS:", len(rows))
print("HUMAN VERIFIED:", sources["HUMAN_VERIFIED"])
print("AI UNVERIFIED:", sources["AI_SUGGESTED_UNVERIFIED"])

for collection in ("BASELINE", "SECTION_AWARE"):
    cases = [
        scores(items)
        for (qid, name), items in groups.items()
        if name == collection and not qid.startswith("OOS")
    ]

    assert len(cases) == 16

    hit = sum(x[0] for x in cases) / len(cases)
    precision = sum(x[1] for x in cases) / len(cases)
    mrr = sum(x[2] for x in cases) / len(cases)

    print(f"\nCOLLECTION: {collection}")
    print("IN-SCOPE QUESTIONS:", len(cases))
    print(f"HIT@4: {hit:.2%}")
    print(f"PRECISION@4: {precision:.2%}")
    print(f"MRR@4: {mrr:.4f}")

    oos = [
        scores(items)
        for (qid, name), items in groups.items()
        if name == collection and qid.startswith("OOS")
    ]

    assert len(oos) == 4
    print("OOS WITH DIRECT MATCH:", sum(x[0] for x in oos), "/ 4")

print("\nEVALUATION: PASS")
print("WARNING: MIXED HUMAN + AI LABELS")
