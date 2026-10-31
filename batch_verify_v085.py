import csv
import subprocess
import sys
from pathlib import Path

path = Path("eval_results/relevance_verified_v084.csv")

if not path.exists():
    raise SystemExit("ERROR: Verified CSV not found")

with path.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

selected_ids = set(sys.argv[1:])
valid_ids = {r["id"] for r in rows}

if selected_ids - valid_ids:
    raise SystemExit("ERROR: Unknown question ID")

for row in rows:
    if selected_ids and row["id"] not in selected_ids:
        continue

    if row["final_relevance"] in {"0", "1", "2"}:
        continue

    print("\n" + "=" * 55)
    print("NEXT:", row["id"])
    print("COLLECTION:", row["collection"])
    print("RANK:", row["rank"])
    print("PAGE:", row["page"])
    print("AI SUGGESTION:", row["suggested_relevance"])

    action = input(
        "\nEnter=review | s=skip | q=quit: "
    ).strip().lower()

    if action == "q":
        break

    if action == "s":
        continue

    if action != "":
        print("INVALID INPUT")
        continue

    subprocess.run([
        sys.executable,
        "verify_label_v084.py",
        row["id"],
        row["collection"],
        row["rank"],
    ])

with path.open(encoding="utf-8-sig") as f:
    updated = list(csv.DictReader(f))

done = sum(
    r["final_relevance"] in {"0", "1", "2"}
    for r in updated
)

print("\n========== FINAL PROGRESS ==========")
print("VERIFIED:", done)
print("REMAINING:", len(updated) - done)
print("TOTAL:", len(updated))
