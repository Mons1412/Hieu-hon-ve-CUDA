# CUDA Knowledge Assistant — Local Docker (Ubuntu Linux)

This is a **local-only** deployment intended for demonstration and testing.
It is not a cloud-ready or public-facing deployment.

## Requirements

- Ubuntu Linux, Docker Engine and Docker Compose plugin.
- Ollama **running on the Ubuntu host** (`http://127.0.0.1:11434`).
- Host models `qwen2.5:3b` and `embeddinggemma:300m` already installed.
- Existing populated `./chroma_db` and source PDF(s) under `./docs/pdf`.

## Run

```bash
ollama list
curl -fsS http://127.0.0.1:11434/api/tags > /dev/null && echo 'OLLAMA: PASS'
docker compose config --quiet
docker compose up --build -d
curl -fsS http://127.0.0.1:8012/health/ready | python -m json.tool
```

- Web UI: `http://127.0.0.1:8012/`
- OpenAPI: `http://127.0.0.1:8012/docs`
- Stop: `docker compose down`
- Logs: `docker compose logs -f --tail=80 cuda-rag`

## Architecture and safety

- `web_ui:app` serves the UI and API on **loopback**, port 8012.
- Compose uses **Linux host networking** so the container can reach the
  existing host Ollama and the original hardcoded readiness URL.
- ChromaDB is **bind mounted** from `./chroma_db` (persistent; never reindexed).
- PDFs are **read-only bind mounted** from `./docs/pdf`.
- The container runs as the host UID/GID set when the scaffold was generated.
- `CUDA_RESPONSE_GATE=0` is the default. Its Bank Conflicts guard is
  experimental and does not certify answer correctness.
- Do **not** publish this host-network service on an externally reachable IP.
- Source PDFs, model weights and the ChromaDB index are **not included** in
  the Docker image or Git repository.

This Linux-local Compose configuration is **not portable** to Docker Desktop,
Windows, macOS or cloud deployments without changes to networking and Ollama
readiness checks.
