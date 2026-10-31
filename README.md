# Hieu-hon-ve-CUDA — CUDA Knowledge Assistant

A **local, document-grounded Retrieval-Augmented Generation (RAG)** assistant for exploring CUDA concepts in NVIDIA technical documentation. Built with Python, FastAPI, ChromaDB, Ollama, and a browser-based chat interface. A **Docker Compose demo for Ubuntu Linux** is available.

> **Project status: local Docker demo v1.0.** The application runs end-to-end on the developer's Ubuntu machine. It is not a public/cloud deployment, and AI-generated explanations are not guaranteed to be technically correct.

## What it does

- Answers CUDA questions in Vietnamese using retrieved passages from a locally indexed technical PDF.
- Shows PDF page references and the retrieved evidence alongside the answer.
- Returns `ABSTAIN` when the model cannot support an answer from the available evidence.
- Exposes health, document-listing, retrieval, and chat endpoints through FastAPI.
- Provides an **experimental**, narrowly scoped Response Gate for a known Bank Conflicts attribution error; **disabled by default**.
- Runs as a local Docker container while reusing the existing **host Ollama models and local ChromaDB index**.

## Architecture

```text
Browser / curl
      |
      v
FastAPI + Web UI  (web_ui:app, localhost:8012)
      |
      +-- Retrieval: query -> EmbeddingGemma -> ChromaDB
      |                                  -> top PDF passages
      |
      +-- Generation: passages + question -> Ollama / Qwen2.5 3B
      |                                         |
      +-- Optional experimental Response Gate <-+
      |
      v
JSON answer + PDF citations + evidence
```

| Component | Implementation |
| --- | --- |
| Application | Python 3.12, FastAPI, Uvicorn |
| Language model | Ollama `qwen2.5:3b` |
| Embedding model | Ollama `embeddinggemma:300m` |
| Vector storage | ChromaDB (persistent local directory) |
| Primary indexed collection | `cuda_pdf_v03`, 177 chunks in the tested environment |
| Source corpus | User-supplied NVIDIA CUDA C++ Best Practices Guide PDF |
| UI | Lightweight web chat in `web_ui.py` |
| Container | Docker Engine + Docker Compose v2 on Ubuntu Linux |

## Quick start — Ubuntu Linux (existing index)

**Requirements:** Docker Engine, Docker Compose v2, Ollama running **on the Ubuntu host** at `127.0.0.1:11434`, and the models below:

```bash
ollama pull qwen2.5:3b
ollama pull embeddinggemma:300m
ollama list
```

The **PDF documents and indexed ChromaDB directory are not bundled with this repository or Docker image**. Before running the demo, provide your own authorized PDF under `docs/pdf/` and an **already populated, compatible** `chroma_db/` directory. This Compose configuration does **not** create or index a database automatically. See `pdf_rag.py` if you are preparing a new collection.

From the repository root:

```bash
docker compose config --quiet
docker compose up --build -d
curl -fsS http://127.0.0.1:8012/health/ready | python -m json.tool
```

Open **http://127.0.0.1:8012/** in a browser. API documentation: **http://127.0.0.1:8012/docs**.

To stop the demo:

```bash
docker compose down
```

### Smoke test

```bash
curl -sS --fail-with-body --max-time 180 \
  -X POST http://127.0.0.1:8012/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Shared Memory giúp tối ưu Matrix Multiplication như thế nào?","top_k":4}' \
  | python -m json.tool
```

A successful response may contain `"status": "ANSWER"` and a `sources` list. **HTTP 200, a citation, and `ANSWER` do not prove factual correctness.**

### Useful diagnostics

```bash
docker compose ps
docker compose logs --tail=60 cuda-rag
curl -fsS http://127.0.0.1:11434/api/tags > /dev/null && echo 'OLLAMA: PASS'
```

If `/health/ready` fails, first check host Ollama, Docker logs, and the mounted `chroma_db/` directory. The application binds to **127.0.0.1**, not a public interface.

## Experimental Response Gate

The API supports `CUDA_RESPONSE_GATE=1` to enable a **limited Bank Conflicts regression guard**. With the flag unset or `0`, the original chat response is returned. The gate can block responses with a known incorrect or ambiguous hardware-actor attribution; it does not validate every CUDA claim.

To test, change `CUDA_RESPONSE_GATE` to `"1"` in `docker-compose.yml`, then run:

```bash
docker compose up -d --force-recreate
```

Revert it to `"0"` to restore the default. Do **not** treat `PASS_THROUGH_UNVERIFIED` as proof of faithfulness.

## Evaluation and limitations

- Retrieval experiment v0.8.8: a 20-question set compared baseline and section-aware indexes at Top-4. **Preliminary, mixed-source labels:** 8 human-verified and 152 AI-suggested/unverified judgments across 160 retrieved rows.
- Preliminary in-scope Precision@4: **42.19% baseline** vs. **48.44% section-aware**; these are **not independently validated production metrics**.
- Faithfulness v0.9: **6/8 expected ANSWER/ABSTAIN statuses matched**; status match is not the same as technical correctness.
- Response Gate v0.9.5: the known Bank Conflicts problem was blocked in a live API test; ordinary Matrix Multiplication answering and out-of-scope abstention also behaved as expected in the small smoke set.
- The tested Docker API still uses the **baseline `cuda_pdf_v03` collection**, not the experimental section-aware collection.
- Only a portion of the source PDF was indexed in the tested environment. The project is **a learning and research prototype**, not a fact-verification system.

See `EVALUATION_REPORT.md`, `PROJECT_CONTEXT.md`, and the evaluation scripts for methodology and historical experiments. Most generated raw evaluation files are intentionally local and excluded by `.gitignore`.

## Local-only deployment and data safety

- Docker Compose uses **`network_mode: host` on Linux** to reach Ollama at `127.0.0.1:11434`; this configuration is **not directly portable** to Docker Desktop, Windows, macOS, or cloud platforms.
- The container runs Uvicorn on **127.0.0.1:8012**. **Do not expose this prototype to public networks**; it does not implement production authentication, rate limiting, or hardening.
- `./chroma_db` is bind-mounted for persistence, and `./docs/pdf` is mounted read-only.
- Model weights, PDFs, vector database contents, evaluation outputs, credentials, and `.env` files are not part of the Docker image or public repository.
- The source PDF belongs to its respective copyright owner; obtain and use it in accordance with its terms.

## Repository entry points

| File | Purpose |
| --- | --- |
| `web_ui.py` | Browser interface and PDF viewing route |
| `api.py` | FastAPI endpoints, retrieval, and answer generation integration |
| `pdf_rag.py` | Original PDF indexing and retrieval logic |
| `rag_answer_v043.py` | Structured Qwen answer generation |
| `api_gate_v095.py` | Experimental API Response Gate integration |
| `subject_guard_v093.py` | Bank Conflicts actor-attribution regression guard |
| `docker-compose.yml` | Ubuntu-only local demo configuration |
| `DOCKER_LOCAL.md` | Detailed local Docker operating notes |

## Roadmap

1. Grow the **human-verified** faithfulness test set and evaluate false blocks.
2. Improve answer grounding for CUDA terminology and hardware-actor attribution.
3. Add a clean-environment indexing/bootstrap path (current Docker demo requires a prepared index).
4. Develop a portable, secured deployment strategy separate from the Linux-local demo.

---

**Repository:** `Mons1412/Hieu-hon-ve-CUDA`  
**Scope:** Educational CUDA documentation assistant; AI output requires verification against source material.
