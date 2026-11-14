# Verify the local Docker demo

The v1.0.0 demo runs on Ubuntu and reuses an Ollama service installed on the host. The PDF and ChromaDB index must already be prepared as described in the project README.

## Preflight

```bash
ollama list
docker compose version
docker compose config --quiet
```

Confirm that `qwen2.5:3b` and `embeddinggemma:300m` are available in Ollama.

## Start and inspect

```bash
docker compose up --build -d
docker compose ps
docker compose logs --tail=40 cuda-rag
```

## Verify readiness

```bash
curl -fsS http://127.0.0.1:8012/health/ready | python3 -m json.tool
```

A healthy response contains `"status": "ready"` and identifies the ChromaDB collection and models. Readiness does not itself prove answer correctness.

## Verify an end-to-end RAG response

```bash
curl -sS --fail-with-body --max-time 180 \
  -X POST http://127.0.0.1:8012/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does shared memory help matrix multiplication?","top_k":4}' \
  | python3 -m json.tool
```

Inspect `status`, `answer` and `sources`. A valid citation format is **not** proof that the answer faithfully reflects the document.

Open <http://127.0.0.1:8012/> to test the web UI from the Ubuntu host.

## Stop

```bash
docker compose down
```

This stops the container without intentionally deleting the host's Ollama models or ChromaDB index.
