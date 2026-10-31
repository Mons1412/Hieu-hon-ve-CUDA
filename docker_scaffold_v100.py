#!/usr/bin/env python3
"""Create a Linux-local Docker scaffold for Mons1412/Hieu-hon-ve-CUDA.

Uses the existing host Ollama via Linux host networking. Does not move or
reindex Chroma data, download models, or edit application Python files.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_FILES = (
    "api.py", "api_gate_v095.py", "rag_answer_v043.py",
    "pdf_rag.py", "web_ui.py",
)
PACKAGES = ("fastapi", "uvicorn", "chromadb", "ollama", "PyMuPDF", "pydantic")
OUTPUTS = ("Dockerfile", "docker-compose.yml", ".dockerignore", "requirements-docker.txt", "DOCKER_LOCAL.md")


def make_files(pyver: str, versions: dict[str, str], uid: int, gid: int) -> dict[str, str]:
    requirements = "# Pinned to the project's currently installed Python environment.\n"
    requirements += "\n".join(f"{p}=={versions[p]}" for p in PACKAGES) + "\n"
    dockerfile = f"""# Linux-local CUDA RAG image; Ollama runs on the Ubuntu host.
FROM python:{pyver}-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update \\
 && apt-get install -y --no-install-recommends libgomp1 \\
 && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt /app/
RUN python -m pip install --no-cache-dir -r requirements-docker.txt
COPY api.py api_gate_v095.py rag_answer_v043.py pdf_rag.py web_ui.py /app/
EXPOSE 8012
CMD ["python", "-m", "uvicorn", "web_ui:app", "--host", "127.0.0.1", "--port", "8012"]
"""
    compose = f"""# Linux-only local setup: host network reuses Ollama at 127.0.0.1:11434.
# Do not publish this service to a public network.
services:
  cuda-rag:
    build:
      context: .
      dockerfile: Dockerfile
    network_mode: host
    user: "{uid}:{gid}"
    environment:
      OLLAMA_HOST: "http://127.0.0.1:11434"
      CUDA_RESPONSE_GATE: "0"  # experimental; use 1 only when intentionally testing
      PYTHONUNBUFFERED: "1"
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./docs/pdf:/app/docs/pdf:ro
    restart: "no"
    command: ["python", "-m", "uvicorn", "web_ui:app", "--host", "127.0.0.1", "--port", "8012"]
"""
    dockerignore = """.git/
.venv/
venv/
__pycache__/
**/__pycache__/
chroma_db/
docs/pdf/
eval_results/
.pytest_cache/
.env
.env.*
*.pdf
*.zip
*.gguf
*.db
*.sqlite
*.sqlite3
*.log
*.pem
*.key
*backup*.py
"""
    guide = """# CUDA Knowledge Assistant — Local Docker (Ubuntu Linux)

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
"""
    return {
        "Dockerfile": dockerfile,
        "docker-compose.yml": compose,
        ".dockerignore": dockerignore,
        "requirements-docker.txt": requirements,
        "DOCKER_LOCAL.md": guide,
    }


def selftest() -> None:
    files = make_files("3.12", {p: "1.2.3" for p in PACKAGES}, 1000, 1000)
    assert set(files) == set(OUTPUTS)
    assert 'FROM python:3.12-slim-bookworm' in files["Dockerfile"]
    assert '"web_ui:app"' in files["Dockerfile"]
    assert 'network_mode: host' in files["docker-compose.yml"]
    assert './chroma_db:/app/chroma_db' in files["docker-compose.yml"]
    assert './docs/pdf:/app/docs/pdf:ro' in files["docker-compose.yml"]
    assert 'CUDA_RESPONSE_GATE: "0"' in files["docker-compose.yml"]
    assert "chroma_db/" in files[".dockerignore"]
    assert all(f"{p}==1.2.3" in files["requirements-docker.txt"] for p in PACKAGES)
    print("DOCKER SCAFFOLD SELFTEST: PASS")


def doctor(root: Path, check_python: bool = True) -> dict[str, str]:
    missing = [name for name in APP_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Missing application files: " + ", ".join(missing))
    if not (root / "chroma_db").is_dir():
        raise RuntimeError("Missing chroma_db/; refusing to generate an empty DB mount")
    if not (root / "docs" / "pdf").is_dir():
        raise RuntimeError("Missing docs/pdf/; refusing to generate an empty PDF mount")
    if not any((root / "docs" / "pdf").rglob("*.pdf")):
        raise RuntimeError("docs/pdf/ has no PDF files")
    code = (root / "api.py").read_text(encoding="utf-8")
    if "from api_gate_v095 import apply_to_api" not in code:
        raise RuntimeError("api.py does not include response gate v0.9.5; check repository version")
    versions = {}
    if check_python:
        for pkg in PACKAGES:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError as e:
                raise RuntimeError(f"Missing package in active Python environment: {pkg}") from e
    docker_bin = shutil.which("docker")
    print("PROJECT CHECK: PASS (application files, PDF, ChromaDB directory, API gate)")
    print("PYTHON:", sys.version.split()[0])
    print("DOCKER:", docker_bin if docker_bin else "NOT FOUND (install Docker before build)")
    if docker_bin:
        try:
            result = subprocess.run(
                [docker_bin, "compose", "version"], capture_output=True,
                text=True, timeout=8, check=False,
            )
            print("COMPOSE:", result.stdout.strip() if result.returncode == 0
                  else "NOT READY: " + result.stderr.strip()[:150])
        except (OSError, subprocess.TimeoutExpired):
            print("COMPOSE: unable to check")
    for p in sorted(versions):
        print("  ", p, versions[p])
    return versions


def write(root: Path, force: bool) -> None:
    versions = doctor(root)
    existing = [x for x in OUTPUTS if (root / x).exists()]
    if existing and not force:
        raise RuntimeError("Output file(s) already exist: " + ", ".join(existing) +
                           ". Nothing was changed. Inspect them first (or use --force intentionally).")
    if not versions:
        raise RuntimeError("No Python package versions found")
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    generated = make_files(pyver, versions, os.getuid(), os.getgid())
    # Inputs verified before writing anything.
    for name, content in generated.items():
        (root / name).write_text(content, encoding="utf-8")
        print("CREATED:", name)
    print("DOCKER SCAFFOLD: PASS")
    print("NEXT: docker compose config --quiet && docker compose up --build -d")
    print("BROWSER: http://127.0.0.1:8012/")
    print("NOTE: Local Linux only, uses host Ollama; CUDA_RESPONSE_GATE defaults OFF.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    p.add_argument("--force", action="store_true", help="Overwrite existing Docker scaffold files intentionally")
    args = p.parse_args()
    try:
        if args.selftest:
            selftest()
        elif args.check:
            doctor(Path.cwd())
        elif args.write:
            write(Path.cwd(), args.force)
    except RuntimeError as error:
        print("STOP:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
