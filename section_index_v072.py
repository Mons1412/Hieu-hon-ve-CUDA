import argparse
import hashlib
import re
from pathlib import Path

import pymupdf

PDF = Path("docs/pdf/CUDA_C_Best_Practices_Guide.pdf")
COLLECTION = "cuda_pdf_section_v072"

WORDS = 140
OVERLAP = 25
BATCH = 10

HEADING = re.compile(
    r"^((?:[1-9]|1[0-9]|20)(?:\.\d+){1,5})\.?\s+"
    r"(?!GB/s\b|MB/s\b|KB/s\b|MHz\b|GHz\b)([A-Za-z].+)$",
    re.IGNORECASE,
)


def make_records():
    section = "unknown"
    title = "Unclassified CUDA material"
    records = []

    with pymupdf.open(PDF) as pdf:
        # Cùng phạm vi 60 trang với baseline.
        # Bỏ trang 1-6 vì chứa bìa/mục lục.
        for page_no in range(7, min(60, len(pdf)) + 1):
            lines = [
                line.strip()
                for line in pdf[page_no - 1].get_text("text").splitlines()
            ]

            segments = []
            buffer = []

            def flush():
                if buffer:
                    segments.append(
                        (section, title, " ".join(buffer))
                    )
                    buffer.clear()

            for i, line in enumerate(lines):
                if not line:
                    continue

                if line.startswith(
                    "CUDA C++ Best Practices Guide, Release"
                ):
                    continue

                # Loại bỏ footer cuối trang.
                if i >= len(lines) - 3 and (
                    line.isdigit()
                    or line.endswith("Device Memory Spaces")
                ):
                    continue

                # Skip printed section footers.
                if i >= len(lines) - 3 and re.match(
                    r"^\d+\.\d+\.\s+[A-Za-z]", line
                ):
                    continue

                match = HEADING.match(line)

                if match:
                    flush()
                    section = match.group(1)
                    title = match.group(2)
                else:
                    buffer.append(line)

            flush()

            for sec, name, body in segments:
                tokens = body.split()

                if len(tokens) < 12:
                    continue

                step = WORDS - OVERLAP

                for part, start in enumerate(
                    range(0, len(tokens), step), 1
                ):
                    end = min(start + WORDS, len(tokens))

                    content = (
                        f"Section {sec}: {name}\n"
                        + " ".join(tokens[start:end])
                    )

                    key = f"{PDF.name}|{page_no}|{sec}|{part}"

                    records.append({
                        "id": hashlib.sha256(
                            key.encode()
                        ).hexdigest()[:32],
                        "text": content,
                        "metadata": {
                            "source": PDF.name,
                            "page": page_no,
                            "section": sec,
                            "section_title": name,
                            "chunk": part
                        }
                    })

                    if end == len(tokens):
                        break

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    records = make_records()

    page53 = [
        r for r in records
        if r["metadata"]["page"] == 53
    ]

    labels = sorted({
        r["metadata"]["section"]
        for r in page53
    })

    print("PAGE 53 SECTIONS:", labels)

    assert "10.2.3.2" in labels
    assert "10.2.3.3" in labels

    print("SECTION SPLIT: PASS")
    print("CANDIDATE CHUNKS:", len(records))

    if not (args.index or args.compare):
        return

    import chromadb
    from ollama import embed

    from pdf_rag import (
        DB_DIR,
        EMBED_MODEL,
        get_collection,
        retrieve
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    if args.index:
        collection = client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=None,
            configuration={
                "hnsw": {"space": "cosine"}
            }
        )

        existing = set(
            collection.get(include=[])["ids"]
        )

        pending = [
            r for r in records
            if r["id"] not in existing
        ]

        print("EXISTING:", len(existing))
        print("PENDING:", len(pending))

        for offset in range(0, len(pending), BATCH):
            batch = pending[offset:offset + BATCH]

            vectors = embed(
                model=EMBED_MODEL,
                input=[r["text"] for r in batch]
            ).embeddings

            collection.upsert(
                ids=[r["id"] for r in batch],
                documents=[r["text"] for r in batch],
                metadatas=[
                    r["metadata"] for r in batch
                ],
                embeddings=[
                    list(v) for v in vectors
                ]
            )

            print(
                f"EMBEDDED: "
                f"{min(offset+BATCH, len(pending))}"
                f"/{len(pending)}",
                flush=True
            )

        print("NEW COLLECTION:", COLLECTION)
        print("NEW CHUNKS:", collection.count())
        print("OLD CHUNKS:", get_collection().count())

    if args.compare:
        old = get_collection()
        new = client.get_collection(COLLECTION)

        questions = {
            "BC01": (
                "Shared Memory có tự động loại bỏ "
                "Bank Conflicts không?"
            ),
            "MM01": (
                "Shared Memory giúp tối ưu "
                "Matrix Multiplication như thế nào?"
            ),
            "MM03": (
                "Tại sao cần tái sử dụng dữ liệu "
                "giữa các thread?"
            )
        }

        for label, question in questions.items():
            a = retrieve(old, question, evidence_limit=4)
            b = retrieve(new, question, evidence_limit=4)

            old_pages = [
                x["metadata"]["page"] for x in a
            ]
            new_pages = [
                x["metadata"]["page"] for x in b
            ]
            sections = [
                x["metadata"].get("section")
                for x in b
            ]

            print(f"\n{label}")
            print("OLD PAGES:", old_pages)
            print("NEW PAGES:", new_pages)
            print("NEW SECTIONS:", sections)

    print("\nDONE")


if __name__ == "__main__":
    main()
