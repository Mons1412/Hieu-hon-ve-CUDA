import csv
from pathlib import Path
import chromadb
from ollama import embed
from pdf_rag import DB_DIR, EMBED_MODEL, get_collection

with open("golden_candidates_v061.csv", encoding="utf-8-sig") as f:
    cases = list(csv.DictReader(f))

client = chromadb.PersistentClient(path=str(DB_DIR))
collections = {
    "BASELINE": get_collection(),
    "SECTION_AWARE": client.get_collection("cuda_pdf_section_v073"),
}

output = Path("eval_results/retrieval_review_v081.csv")
output.parent.mkdir(exist_ok=True)
total = 0

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "question", "collection", "rank", "page",
        "section", "distance", "evidence", "relevance", "notes"
    ])

    for case in cases:
        question = case["question"]
        vector = list(embed(model=EMBED_MODEL, input=question).embeddings[0])

        print(f'\n{case["id"]}:', flush=True)

        for name, collection in collections.items():
            result = collection.query(
                query_embeddings=[vector],
                n_results=4,
                where={"page": {"$gte": 7}},
                include=["documents", "metadatas", "distances"]
            )

            docs = result["documents"][0]
            metas = result["metadatas"][0]
            distances = result["distances"][0]

            assert len(docs) == 4, f"Missing results: {case['id']} {name}"

            for rank, (doc, meta, dist) in enumerate(
                zip(docs, metas, distances), 1
            ):
                writer.writerow([
                    case["id"], question, name, rank,
                    meta["page"], meta.get("section", ""),
                    round(float(dist), 5),
                    " ".join(doc.split())[:800], "", ""
                ])
                total += 1

            print(name, [m["page"] for m in metas], flush=True)

assert len(cases) == 20
assert total == 160

print("\nFAIR-SCOPE BENCHMARK: PASS")
print("QUESTIONS:", len(cases))
print("RESULTS:", total)
print("OUTPUT:", output)
