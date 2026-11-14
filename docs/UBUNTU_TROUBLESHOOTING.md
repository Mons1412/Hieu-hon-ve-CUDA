# Ubuntu local deployment troubleshooting

These notes describe issues encountered while packaging the v1.0.0 local Docker demo. They assume the project repository has been cloned and the required PDF and ChromaDB data have been prepared.

## `docker: unknown command: docker compose`

The Docker CLI is present, but the Compose v2 plugin may be missing.

On Ubuntu 24.04, first check the available package and install it if present:

```bash
sudo apt update
apt-cache policy docker-compose-v2
sudo apt install docker-compose-v2
docker compose version
```

Use the Docker-maintained Compose plugin instructions if your Docker package source differs from Ubuntu's.

## `curl: (7) Failed to connect` on port 8012

Confirm the container is running and inspect logs:

```bash
docker compose ps
docker compose logs --tail=100 cuda-rag
```

Check the configured port and networking in `docker-compose.yml`. The local Ubuntu setup is not intended to be a public internet deployment.

## Ollama models are missing

```bash
ollama list
ollama pull qwen2.5:3b
ollama pull embeddinggemma:300m
```

Make sure Ollama is running on the Ubuntu host and reachable under the network arrangement documented in `DOCKER_LOCAL.md`. A healthy Docker process alone does not guarantee the host Ollama service is reachable.

## Readiness fails because the collection is missing

The demo depends on an existing PDF-derived ChromaDB index. Verify your data directory and indexing steps rather than assuming Docker creates the index automatically. Do not commit the ChromaDB database, model weights or environment secrets to Git.

## `favicon.ico` returns 404

An absent favicon is cosmetic. Confirm the main page (`/`) and API (`/api/v1/chat`) work instead.

## Need to inspect a failing response

```bash
docker compose logs -f cuda-rag
```

Capture the HTTP status and JSON body; remove tokens, secrets and personal information before sharing logs.
