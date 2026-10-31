#!/usr/bin/env python3
"""CUDA Knowledge Assistant v0.4 — local FastAPI, reusing pdf_rag.py v0.3.3.

Read-only API: no PDF uploads, index changes, or DB deletion.
Run from the project root after copying to api.py:
    uvicorn api:app --host 127.0.0.1 --port 8000
"""

from collections import defaultdict
from functools import lru_cache
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ollama import chat

from pdf_rag import (
    ABSTAIN_TEXT,
    COLLECTION_NAME,
    LLM_MODEL,
    EMBED_MODEL,
    get_collection,
    retrieve,
    validate_answer,
)

app = FastAPI(
    title="CUDA Knowledge Assistant API",
    description="Read-only local RAG API backed by ChromaDB, EmbeddingGemma and Gemma 3 1B.",
    version="0.4.3",
)

# Development UI origins only; API is additionally bound to localhost in the run command.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=3000)
    top_k: int = Field(default=4, ge=1, le=4)


@lru_cache(maxsize=1)
def collection():
    return get_collection()


def require_collection():
    try:
        db = collection()
        if db.count() == 0:
            raise HTTPException(
                status_code=503,
                detail="PDF collection is empty. Index PDFs explicitly using pdf_rag.py.",
            )
        return db
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ChromaDB unavailable: {type(exc).__name__}") from exc


def search_evidence(question: str, top_k: int):
    db = require_collection()
    try:
        evidence = retrieve(db, question, evidence_limit=top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding or retrieval failed: {type(exc).__name__}: {exc}",
        ) from exc

    return [
        {
            "rank": i,
            "source": item["metadata"]["source"],
            "page": int(item["metadata"]["page"]),
            "chunk": int(item["metadata"].get("chunk", 0)),
            "distance": round(float(item["distance"]), 4),
            "preview": " ".join(item["document"].split())[:260],
            "text": item["document"],
        }
        for i, item in enumerate(evidence, 1)
    ]



def generate_answer(question: str, sources: list[dict]):
    import re
    from typing import Literal
    from pydantic import BaseModel, Field, ValidationError

    class Claim(BaseModel):
        text: str = Field(min_length=6, max_length=350)
        source_id: int = Field(ge=1)
        quote: str = Field(min_length=25, max_length=240)

    class Draft(BaseModel):
        status: Literal["ANSWER", "ABSTAIN"]
        claims: list[Claim] = Field(max_length=2)

    def normalize(value):
        return " ".join(value.split()).casefold()

    evidence = "\n\n".join(
        f"SOURCE [{s['rank']}] | PDF page {s['page']}\n{s['text']}"
        for s in sources
    )

    prompt = (
        "Answer using EVIDENCE only. Output JSON matching schema. "
        "If evidence directly supports an answer, prefer ANSWER. "
        "For yes/no questions, begin with Có or Không. "
        "Write 1-2 short Vietnamese claims. "
        "Each claim requires ONE source_id and ONE exact English "
        "quote copied from that source. "
        "Do not combine facts from different sources in one claim. "
        "If no direct supporting quote exists, use ABSTAIN "
        "with claims=[]. Do not invent facts.\n\n"
        f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}"
    )

    last_error = "no validated claims"

    for attempt in range(2):
        extra = (
            "\nRecheck evidence before abstaining. "
            f"Previous issue: {last_error}"
            if attempt else ""
        )

        response = chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful CUDA researcher. JSON only."
                },
                {
                    "role": "user",
                    "content": prompt + extra
                }
            ],
            format=Draft.model_json_schema(),
            options={"temperature": 0, "num_predict": 450}
        )

        try:
            draft = Draft.model_validate_json(
                response.message.content
            )
        except (ValidationError, ValueError):
            last_error = "invalid structured JSON"
            continue

        if draft.status == "ABSTAIN":
            if not draft.claims and attempt == 1:
                return (
                    "ABSTAIN",
                    ABSTAIN_TEXT,
                    "model could not find supporting evidence"
                )
            last_error = "model abstained; recheck evidence"
            continue

        if not draft.claims:
            last_error = "ANSWER without claims"
            continue

        lines = []

        for item in draft.claims:
            if item.source_id > len(sources):
                last_error = "invalid source ID"
                break

            source = sources[item.source_id - 1]
            quote = " ".join(item.quote.split())

            if (
                len(quote.split()) < 6
                or normalize(quote) not in normalize(source["text"])
            ):
                last_error = "quote not found in cited PDF chunk"
                break

            claim = " ".join(item.text.split()).rstrip(". ")

            if re.search(r"\[\d+\]", claim):
                last_error = "claim contains citation markers"
                break

            sid = item.source_id

            lines.append(
                f"{claim}. [{sid}] | "
                f"Bằng chứng gốc: “{quote}” [{sid}]"
            )

        else:
            answer = "\n".join(lines)

            state, reason = validate_answer(
                answer, len(sources)
            )

            if state == "ANSWER":
                return (
                    "ANSWER",
                    answer,
                    "PDF quote matched; meaning not independently verified"
                )

            last_error = reason

    return "INVALID", None, last_error

from rag_answer_v043 import generate_answer

@app.get("/health/live")
def liveness():
    return {"status": "ok", "version": app.version}


@app.get("/health/ready")
def readiness():
    db = require_collection()
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=4) as response:
            if response.status != 200:
                raise URLError("Ollama did not return HTTP 200")
            # Only verify server responsiveness; model list can be checked separately.
    except (URLError, OSError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Ollama server is unavailable") from exc
    return {
        "status": "ready",
        "collection": COLLECTION_NAME,
        "chunks": db.count(),
        "llm": LLM_MODEL,
        "embedding_model": EMBED_MODEL,
    }


@app.get("/api/v1/documents")
def list_documents():
    db = require_collection()
    count = db.count()
    grouped = defaultdict(lambda: {"pages": set(), "chunks": 0})
    for offset in range(0, count, 500):
        result = db.get(include=["metadatas"], limit=500, offset=offset)
        for meta in result["metadatas"] or []:
            if meta is None:
                continue
            source = meta.get("source", "unknown")
            grouped[source]["chunks"] += 1
            if "page" in meta:
                grouped[source]["pages"].add(int(meta["page"]))
    return {
        "collection": COLLECTION_NAME,
        "total_chunks": count,
        "documents": [
            {"source": name, "chunks": info["chunks"], "indexed_pages": len(info["pages"])}
            for name, info in sorted(grouped.items())
        ],
    }


@app.get("/api/v1/search")
def search(question: str = Query(min_length=3, max_length=3000), top_k: int = Query(4, ge=1, le=4)):
    question = " ".join(question.split())
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="Question is empty")
    sources = search_evidence(question, top_k)
    return {"question": question, "sources": sources, "count": len(sources)}


@app.post("/api/v1/chat")
def chat_endpoint(request: ChatRequest):
    question = " ".join(request.question.split())
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="Question is empty")
    sources = search_evidence(question, request.top_k)
    if not sources:
        return {
            "question": question,
            "status": "ABSTAIN",
            "answer": ABSTAIN_TEXT,
            "reason": "no retrievable PDF passages",
            "sources": [],
            "format_validated": False,
            "fact_verified": False,
        }
    try:
        state, answer, reason = generate_answer(question, sources)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama inference failed: {type(exc).__name__}: {exc}") from exc
    return {
        "question": question,
        "status": state,
        "answer": answer,
        "reason": reason,
        "sources": sources,
        "format_validated": state == "ANSWER",
        "fact_verified": False,
    }
