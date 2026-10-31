# Linux-local CUDA RAG image; Ollama runs on the Ubuntu host.
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt /app/
RUN python -m pip install --no-cache-dir -r requirements-docker.txt
COPY api.py api_gate_v095.py rag_answer_v043.py pdf_rag.py web_ui.py /app/
EXPOSE 8012
CMD ["python", "-m", "uvicorn", "web_ui:app", "--host", "127.0.0.1", "--port", "8012"]
