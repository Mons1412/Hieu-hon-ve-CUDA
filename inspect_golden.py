import argparse
import csv

from api import search_evidence

parser = argparse.ArgumentParser()
parser.add_argument(
    "ids",
    nargs="*",
    default=["BC01", "MM01", "MM03"]
)
parser.add_argument("--chars", type=int, default=850)

args = parser.parse_args()

with open(
    "golden_candidates_v061.csv",
    encoding="utf-8-sig"
) as f:
    cases = {
        row["id"]: row
        for row in csv.DictReader(f)
    }

for case_id in args.ids:
    case = cases.get(case_id)

    if case is None:
        print(f"UNKNOWN CASE: {case_id}")
        continue

    print("\n" + "=" * 65)
    print("CASE:", case_id)
    print("QUESTION:", case["question"])
    print("PROPOSED STATUS:", case["proposed_status"])
    print("VERIFIED:", case["verified"])

    sources = search_evidence(case["question"], 4)

    for source in sources:
        text = " ".join(source["text"].split())

        print("\n" + "-" * 50)
        print("RANK:", source["rank"])
        print("PAGE:", source["page"])
        print("DISTANCE:", source["distance"])
        print("EVIDENCE:")
        print(text[:args.chars])

        if len(text) > args.chars:
            print("... [TRUNCATED]")

print("\nINSPECTION COMPLETED")
print("No gold labels were changed.")
