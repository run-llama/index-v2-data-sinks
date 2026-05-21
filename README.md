# index-v2-data-sinks

Data sink examples for **LlamaCloud Index v2**.

This package shows how to take the output of an Index v2 parse sync
(`ParseSyncOutput`), chunk it, attach provenance metadata, embed it, and
export the resulting chunks to a downstream sink. Reference
implementations are provided for **MongoDB** (Atlas vector search),
**PostgreSQL + pgvector**, and **Turbopuffer**. New sinks can be added by
implementing the shared `Exporter` protocol in `export/base.py`.

## What it does

Given a directory in LlamaCloud, the pipeline will:

1. List all files in the directory and download their `ParseSyncOutputV1`
   JSON payloads from LlamaCloud.
2. Concatenate the per-page markdown (`header` + `text` + `footer`) while
   tracking page boundaries by character offset.
3. Chunk the concatenated text with [chonkie](https://github.com/chonkie-ai/chonkie)
   (`fast`, `sentence`, or `token` strategy).
4. Embed the chunks with the embedding model of choice.
5. For each chunk, build an `ExportedOutputV1` record carrying the chunk
   text, character/token offsets, page range, tenant/config IDs, user
   metadata, attachments, and content-change-detection fields
   (`content_fingerprint_hash`, `source_modified_at`, `exported_at`).
6. Hand the records to an exporter that writes them to the sink. All
   exporters use a delete-then-insert strategy keyed on
   `(parsed_directory_file_id, export_config_id)` so re-parsed files with
   a different chunk count are fully replaced rather than duplicated.

Exporters also expose a `list_snapshots()` method returning lightweight
per-file `ExportSnapshot` records (metadata hash, content fingerprint
hash, timestamps) so an orchestrator can diff against the current sync
directory state and decide what to re-export or delete.

## Project layout

```
src/index_v2_data_sinks/
├── cli.py               # `index-exp` Typer CLI
├── pipeline.py          # End-to-end export_pipeline()
├── parsed_outputs.py    # LlamaCloud directory + file fetching
├── chunking.py          # ChunkerService wrapping chonkie
├── schema.py            # ParseSyncOutput, Chunk, ExportedOutput, PageBoundary
├── embeddings.py        # Embedding model types
├── utils.py             # Hashing helpers (metadata fingerprints, etc.)
├── configs/
│   ├── mongo.py         # MongoDBIndexV2Settings (pydantic-settings)
│   ├── pgvector.py      # PGVectorIndexV2Settings (pydantic-settings)
│   └── turbopuffer.py   # TurbopufferIndexV2Settings (pydantic-settings)
├── services/
│   ├── mongo.py         # MongoService – connection + collection management
│   ├── pgvector.py      # PGVectorService – engine + table/index management
│   └── turbopuffer.py   # TurbopufferService – namespace management
└── export/
    ├── base.py          # Exporter protocol (shared contract)
    ├── snapshot.py      # ExportSnapshot – target-agnostic diff record
    ├── mongo.py         # MongoExporter
    ├── pgvector.py      # PGVectorExporter
    └── turbopuffer.py   # TurbopufferExporter
```

## Requirements

- Python `>=3.13`
- A LlamaCloud API key
- One of:
  - A MongoDB deployment (for the Mongo sink),
  - A PostgreSQL instance with the `vector` extension available (for the
    pgvector sink). See [Local Postgres](#local-postgres-via-docker-compose), or
  - A [Turbopuffer](https://turbopuffer.com) account (for the Turbopuffer sink).

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and declared
in `pyproject.toml`.

## Installation

```bash
uv sync
```

## Configuration

Configuration is read from environment variables (a `.env` file in the
project root is supported via `pydantic-settings`).

### LlamaCloud

| Variable | Description |
| --- | --- |
| `LLAMA_CLOUD_API_KEY` | API key for LlamaCloud. |
| `LLAMA_CLOUD_BASE_URL` | Optional override of the LlamaCloud base URL. |

### MongoDB sink

All variables use the prefix `DEFAULT_INDEX_MONGO_`:

| Variable | Description |
| --- | --- |
| `DEFAULT_INDEX_MONGO_URI` | MongoDB connection URI. |
| `DEFAULT_INDEX_MONGO_DB` | Database name. |
| `DEFAULT_INDEX_MONGO_COLLECTION` | Collection name. |
| `DEFAULT_INDEX_MONGO_URI_MAP` | Optional namespace → URI map (JSON). |
| `DEFAULT_INDEX_MONGO_DB_MAP` | Optional namespace → db-name map (JSON). |
| `DEFAULT_INDEX_MONGO_COLLECTIONS_MAP` | Optional namespace → collection map (JSON). |

### PGVector sink

All variables use the prefix `DEFAULT_INDEX_PGVECTOR_`:

| Variable | Description |
| --- | --- |
| `DEFAULT_INDEX_PGVECTOR_URI` | PostgreSQL connection URI. `postgres://…` and `postgresql://…` are auto-rewritten to `postgresql+asyncpg://…`. |
| `DEFAULT_INDEX_PGVECTOR_TABLE` | Base table name for exported chunks. |
| `DEFAULT_INDEX_PGVECTOR_SCHEMA_NAME` | Schema name. Defaults to `public`. |
| `DEFAULT_INDEX_PGVECTOR_URI_MAP` | Optional namespace → URI map (JSON). |
| `DEFAULT_INDEX_PGVECTOR_SCHEMA_MAP` | Optional namespace → schema map (JSON). |
| `DEFAULT_INDEX_PGVECTOR_TABLES_MAP` | Optional namespace → table-name map (JSON). |

The exporter automatically:

- enables the `vector` extension (`CREATE EXTENSION IF NOT EXISTS vector`),
- creates the schema if missing,
- creates one table per embedding model (the default model keeps the base
  table name; other models get a sanitized suffix) with a properly-sized
  `VECTOR(dims)` column,
- creates an HNSW index on `embedding` (`vector_cosine_ops`) and a
  composite btree on `(export_config_id, parsed_directory_file_id)` to
  back snapshot/diff queries.

### Turbopuffer sink

All variables use the prefix `DEFAULT_INDEX_TPUF_`:

| Variable | Description |
| --- | --- |
| `DEFAULT_INDEX_TPUF_API_KEY` | Platform Turbopuffer API key. |
| `DEFAULT_INDEX_TPUF_REGION` | Default region for namespaces created by the platform. Defaults to `aws-us-east-1`. |
| `DEFAULT_INDEX_TPUF_NAMESPACE_PREFIX` | Prefix prepended to every platform-managed namespace name (final namespace = `{prefix}-{target_id}`). Defaults to `llamacloud`. |
| `DEFAULT_INDEX_TPUF_REGION_MAP` | Optional per-target-id region overrides (JSON). |
| `DEFAULT_INDEX_TPUF_NAMESPACE_MAP` | Optional per-target-id namespace overrides (JSON). Values bypass `namespace_prefix`. |

Each export config gets its own Turbopuffer namespace, derived from the
export config ID and the configured prefix, so a single Turbopuffer
account can host an arbitrary number of tenants without per-tenant
infrastructure.

#### Local Postgres via docker-compose

`compose.yaml` provisions a local pgvector-enabled Postgres:

```yaml
services:
  pgvector:
    image: pgvector/pgvector:pg18
    ports:
      - 9876:5432
    ...
```

> The official `postgres` image does **not** ship the `vector` extension —
> use the `pgvector/pgvector` image (or install the extension yourself)
> otherwise the exporter will fail with
> `extension "vector" is not available`.

Start it with:

```bash
docker compose up -d
```

## Usage

> _The pipeline limits concurrent per-file exports via an
> `asyncio.Semaphore` (`MAX_CONCURRENT_EXPORTS = 10`)._

### CLI

The package installs an `index-exp` entrypoint:

```bash
index-exp \
    --directory-name "UFO Files" \
    --project-id   "your-project-id" \
    --config-id    "your-config-id" \
    --export-to    postgres \
    --embedding-model "openai:text-embedding-3-small" \
    --chunking-strategy sentence \
    --metadata '{"source": "UFO files", "file_types": "pdf"}'
```

A wrapper invocation is also provided in `run-cli.sh`. Run `index-exp --help`
for the full option list.

| Option | Description |
| --- | --- |
| `--directory-name`, `-d` | LlamaCloud directory name. |
| `--project-id`, `-p` | LlamaCloud project ID. |
| `--config-id`, `-c` | Export config ID. |
| `--export-to`, `-t` | `mongo` (default), `postgres`, or `turbopuffer`. |
| `--chunking-strategy`, `-s` | `fast` \| `sentence` \| `token`. Default `sentence`. |
| `--embedding-model`, `-e` | Embedding model id (e.g. `openai:text-embedding-3-small`). |
| `--metadata`, `-m` | JSON-encoded dict attached to every chunk. |

### Programmatic

```python
import asyncio

from index_v2_data_sinks.pipeline import export_pipeline


asyncio.run(
    export_pipeline(
        export_to="postgres",          # or "mongo" | "turbopuffer"
        directory_name="my-index-directory",
        project_id="my-project-id",
        config_id="my-export-config-id",
        chunking_strategy="sentence",  # "fast" | "sentence" | "token"
        embedding_model="openai:text-embedding-3-small",
        metadata={"source": "docs", "tags": ["public", "v1"]},
    )
)
```

## Adding a new sink

Implement the `Exporter` protocol from `export/base.py`:

```python
from index_v2_data_sinks.export.base import Exporter

class MySinkExporter(Exporter):
    async def export(self, outputs, *, project_id, embeddings=None, embedding_model=None): ...
    async def delete_file(self, *, parsed_directory_file_id, export_config_id): ...
    async def delete_files(self, *, parsed_directory_file_ids, export_config_id): ...
    async def list_snapshots(self, *, export_config_id): ...
```

Then wire it into `pipeline.export_pipeline` (and the CLI `--export-to`
choices) alongside the existing sinks.
