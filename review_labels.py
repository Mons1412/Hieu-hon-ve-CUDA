import csv
import sys
from pathlib import Path

path = Path("eval_results/relevance_preliminary_v082.csv")
ids = set(sys.argv[1:])

if not ids:
    raise SystemExit("Usage: python review_labels.py SM01 SM02")

with path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    rows = list(reader)

known = {r["id"] for r in rows}
if ids - known:
    raise SystemExit(f"Unknown IDs: {sorted(ids - known)}")

for row in rows:
    if row["id"] not in ids or row["relevance"].strip():
        continue

    print("\n" + "=" * 60)
    print("CASE:", row["id"])
    print("COLLECTION:", row["collection"])
    print("RANK:", row["rank"], "PAGE:", row["page"])
    print("SECTION:", row["section"])
    print("EVIDENCE:", row["evidence"])

    score = input("Score [0/1/2], s=skip, q=quit: ").strip().lower()

    if score == "q":
        break
    if score not in ("0", "1", "2"):
        continue

    row["relevance"] = score
    row["notes"] = "PRELIMINARY - verify full PDF evidence"

    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    temp.replace(path)
    print("SAVED")

complete = sum(bool(r["relevance"].strip()) for r in rows)
print(f"\nPROGRESS: {complete}/{len(rows)} preliminary labels")
