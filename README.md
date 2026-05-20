# index-v2-data-sinks

Data sink examples for **LlamaCloud Index v2**.

This package shows how to take the output of an Index v2 parse sync
(`ParseSyncOutput`), chunk it, attach provenance metadata, and export the
resulting chunks to a downstream sink. The current reference implementation
targets **MongoDB** (with vector embeddings), but the abstractions are
designed so additional sinks (e.g. Postgres/pgvector, OpenSearch, S3) can
be added by implementing the same exporter interface.

## What it does

Given a directory in LlamaCloud, the pipeline will:

1. List all files in the directory and download their `ParseSyncOutputV1`
   JSON payloads from LlamaCloud.
2. Concatenate the per-page markdown (`header` + `text` + `footer`) while
   tracking page boundaries by character offset.
3. Chunk the concatenated text with [chonkie](https://github.com/chonkie-ai/chonkie)
   (`fast`, `sentence`, or `token` strategy).
4. Embed with the embedding model of choice
4. For each chunk, build an `ExportedOutputV1` record that carries the
   chunk text, character/token offsets, page range, tenant/config IDs,
   user metadata, attachments, and content-change-detection fields
   (`content_fingerprint_hash`, `source_modified_at`, `exported_at`).
5. Hand the records to an exporter that writes them to the sink. The
   MongoDB exporter uses a delete-then-insert strategy keyed on
   `(parsed_directory_file_id, export_config_id)` so re-parsed files with
   a different chunk count are fully replaced rather than duplicated.

Exporters also expose a `list_snapshots()` method returning lightweight
per-file `ExportSnapshot` records (metadata hash, content fingerprint
hash, timestamps) so an orchestrator can diff against the current sync
directory state and decide what to re-export or delete.

## Project layout

```
src/index_v2_data_sinks/
├── pipeline.py          # End-to-end mongo_export_pipeline()
├── parsed_outputs.py    # LlamaCloud directory + file fetching
├── chunking.py          # ChunkerService wrapping chonkie
├── schema.py            # ParseSyncOutput, Chunk, ExportedOutput, PageBoundary
├── embeddings.py        # Embedding model types
├── utils.py             # Hashing helpers (metadata fingerprints, etc.)
├── configs/
│   └── mongo.py         # MongoDBIndexV2Settings (pydantic-settings)
├── services/
│   └── mongo.py         # MongoService – connection + collection management
└── export/
    ├── snapshot.py      # ExportSnapshot – target-agnostic diff record
    └── mongo.py         # MongoExporter – write/delete/list_snapshots
```

## Requirements

- Python `>=3.13`
- A LlamaCloud API key
- A MongoDB deployment (for the Mongo sink)

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

### Postgres sink

> WORK IN PROGRESS

## Usage

> _The pipeline limits concurrent per-file exports via an `asyncio.Semaphore`
(`MAX_CONCURRENT_EXPORTS = 10`)._

### MongoDB

```python
import asyncio

from index_v2_data_sinks.pipeline import mongo_export_pipeline


asyncio.run(
    mongo_export_pipeline(
        directory_name="my-index-directory",
        project_id="my-project-id",
        config_id="my-export-config-id",
        chunking_strategy="sentence",  # "fast" | "sentence" | "token"
        metadata={"source": "docs", "tags": ["public", "v1"]},
    )
)
```
