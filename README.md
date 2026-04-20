# TAU-KG

TAU-KG is a Streamlit-based biomedical knowledge graph project.

## Run locally (without Docker)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run streamlit_app.py
```

## Docker containers

### Standard container (port 8778)

Build and start using the main Compose file:

```bash
docker compose -f compose.yml up --build -d
```

Stop it:

```bash
docker compose -f compose.yml down
```

### Development container (port 8998)

Build and start the development container (uses `Dockerfile.dev` and mounts the repo for active development):

```bash
docker compose -f compose.dev.yml up --build -d
```

Stop it:

```bash
docker compose -f compose.dev.yml down
```

## Open in browser

- Standard app: http://localhost:8778
- Development app: http://localhost:8998

## Paper save mode

Set `PAPER_SAVE_MODE` in `.env`:

- `UPSERT` (default): keep existing paper entries and merge in newly extracted entities/relationships
- `OVERWRITE`: replace any existing paper data for the same paper ID/PMCID

## Chroma DB path

If you see `attempt to write a readonly database` in Docker, set a writable path in `.env`:

- `CHROMA_DB_PATH=/tmp/chroma_db`
