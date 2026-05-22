"""Pinecone export target implementation.

Writes :class:`ExportedOutput` chunks to a Pinecone index. Multi-tenancy
shape:

- one **index** per (target_id, embedding_model) — required because
  Pinecone indexes are dimension-locked;
- one **namespace** per ``export_config_id`` within that index —
  Pinecone namespaces are first-class isolation primitives, free to
  create, and back the cleanest delete / list / snapshot story.

IDs are deterministic ``"{parsed_directory_file_id}#{chunk_index}"``
strings so re-exports are idempotent upserts and per-file cleanup can
use Pinecone's prefix-based ID listing (``list_paginated(prefix=...)``).
This matters because Pinecone *serverless* does not support
delete-by-metadata-filter; only delete-by-id and delete-all-in-namespace
are available, so deletes must know the exact ids to remove.
"""

import functools
import json
from datetime import datetime
from typing import Any

from pinecone.async_client.async_index import AsyncIndex

from ..embeddings import SupportedEmbeddingModel
from ..schema import ExportedOutput, MetadataValue
from ..services.pinecone import (
    DEFAULT_TARGET_ID,
    EXPORT_CONFIG_ID_FIELD,
    METADATA_FIELDS_HASH_FIELD,
    PARSED_DIRECTORY_FILE_ID_FIELD,
    TENANT_ID_FIELD,
    PineconeService,
    get_pinecone_service,
)
from ..utils import compute_metadata_hash
from .base import Exporter
from .snapshot import ExportSnapshot

# Prefix marker between file id and chunk index in a Pinecone vector id.
# Kept as a module constant so the delete path's prefix construction and
# the snapshot path's id parsing can't drift apart.
_ID_SEP = "#"

# Pagination caps for snapshot/cleanup id enumeration. Pinecone's
# ``list_paginated`` returns up to ~100 ids per page on serverless; we
# bound the total pages so a runaway namespace can't stall a snapshot
# scan indefinitely (combined ceiling = _LIST_PAGE_LIMIT * _LIST_MAX_PAGES
# ids per call).
_LIST_PAGE_LIMIT = 100
_LIST_MAX_PAGES = 1000

# Pinecone's ``fetch`` endpoint accepts up to 1000 ids per call; we batch
# at this size when resolving snapshot representatives.
_FETCH_BATCH_SIZE = 1000

# Custom metadata keys are namespaced under ``mf__`` to keep them clearly
# separated from system fields (mirrors the turbopuffer exporter).
_METADATA_FIELD_PREFIX = "mf__"


def _point_id(output: ExportedOutput) -> str:
    """Deterministic Pinecone vector id for a chunk.

    Uses ``{file_id}#{chunk_index}`` so:
    - re-exports of the same chunk overwrite (idempotent upsert), and
    - all chunks of a given file share a common prefix usable with
      Pinecone's ``list_paginated(prefix=...)`` for delete-by-prefix.
    """
    return f"{output.parsed_directory_file_id}{_ID_SEP}{output.chunk_index}"


def _file_id_prefix(parsed_directory_file_id: str) -> str:
    """Prefix used to enumerate all chunks of a single file."""
    return f"{parsed_directory_file_id}{_ID_SEP}"


def _split_id(vector_id: str) -> tuple[str, str | None]:
    """Inverse of :func:`_point_id`: return ``(file_id, chunk_index)``.

    Returns ``chunk_index = None`` if the id does not match the expected
    schema, so the snapshot loop can skip foreign ids that may have been
    written into the namespace by other tooling instead of crashing.
    """
    file_id, sep, chunk = vector_id.partition(_ID_SEP)
    if not sep:
        return vector_id, None
    return file_id, chunk


def _coerce_metadata_value(value: MetadataValue) -> Any:
    """Coerce one custom metadata value to a Pinecone-acceptable type.

    Pinecone metadata values must be ``str``, number, ``bool``, or
    ``list[str]``. ``None`` is rejected by the server, so we drop those
    at write time.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        # ``MetadataListValue`` is ``list[str]``, but be defensive against
        # stringly-typed lists that slipped past the type checker.
        return [str(v) for v in value]


def _build_metadata(output: ExportedOutput) -> dict[str, Any]:
    """Project an ExportedOutput into a Pinecone metadata dict.

    Pinecone allows only flat ``str | number | bool | list[str]`` values,
    so nested fields (``attachments``, ``metadata_fields``) are handled
    explicitly:

    - ``attachments`` is JSON-encoded into ``attachments_json`` when
      non-empty (callers needing structured access can json.loads it);
    - ``metadata_fields`` entries are promoted to top-level keys under a
      ``mf__`` prefix, with values coerced to Pinecone-acceptable types;
    - ``metadata_fields_hash`` is precomputed at write time so
      :meth:`list_snapshots` doesn't have to re-hash a round-tripped
      payload.

    Raises:
        ValueError: if ``tenant_id`` or ``export_config_id`` is empty on
            the row. Both are required for snapshot/delete scoping.
    """
    md: dict[str, Any] = {
        TENANT_ID_FIELD: output.tenant_id,
        EXPORT_CONFIG_ID_FIELD: output.export_config_id,
        PARSED_DIRECTORY_FILE_ID_FIELD: output.parsed_directory_file_id,
        "chunk_index": output.chunk_index,
        "page_range_start": output.page_range_start,
        "page_range_end": output.page_range_end,
        # ``content`` is duplicated into metadata so RAG callers reading
        # query results don't need a second hop to fetch chunk text. This
        # mirrors the turbopuffer attribute model.
        "content": output.content,
    }
    if output.source_file_id is not None:
        md["source_file_id"] = output.source_file_id
    if output.chunk_start_char is not None:
        md["chunk_start_char"] = output.chunk_start_char
    if output.chunk_end_char is not None:
        md["chunk_end_char"] = output.chunk_end_char
    if output.chunk_token_count is not None:
        md["chunk_token_count"] = output.chunk_token_count
    if output.content_fingerprint_hash is not None:
        md["content_fingerprint_hash"] = output.content_fingerprint_hash
    if output.source_modified_at is not None:
        md["source_modified_at"] = output.source_modified_at.isoformat()
    if output.exported_at is not None:
        md["exported_at"] = output.exported_at.isoformat()

    if output.metadata_fields:
        meta_hash = compute_metadata_hash(output.metadata_fields)
        if meta_hash is not None:
            md[METADATA_FIELDS_HASH_FIELD] = meta_hash
        for key, value in output.metadata_fields.items():
            coerced = _coerce_metadata_value(value)
            if coerced is None:
                continue
            md[f"{_METADATA_FIELD_PREFIX}{key}"] = coerced

    if output.attachments:
        md["attachments_json"] = json.dumps(
            [a.model_dump(mode="json") for a in output.attachments]
        )

    return md


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class PineconeExporter(Exporter):
    """Writes ExportedOutput chunks to a Pinecone serverless index."""

    def __init__(
        self,
        *,
        target_id: str = DEFAULT_TARGET_ID,
        target_service: PineconeService | None = None,
    ) -> None:
        self._target_service = target_service or get_pinecone_service()
        self._target_id = target_id

    async def export(
        self,
        outputs: list[ExportedOutput],
        *,
        project_id: str,
        embeddings: list[list[float]] | None = None,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> None:
        if embeddings is None:
            raise ValueError("`embeddings` must be provided for Pinecone export")
        if not outputs:
            return

        index, _ = await self._target_service.ensure_index(
            self._target_id, embedding_model
        )

        # All outputs in a single export() call belong to the same file+config.
        parsed_directory_file_id = outputs[0].parsed_directory_file_id
        export_config_id = outputs[0].export_config_id

        # Replace previous chunks for this (config, file) pair. Pinecone
        # serverless doesn't accept delete-by-metadata-filter, so we list
        # ids by prefix and delete by id. The deterministic id scheme makes
        # this cheap: one list call per file in the typical case.
        stale_ids = await self._list_ids_with_prefix(
            index=index,
            namespace=export_config_id,
            prefix=_file_id_prefix(parsed_directory_file_id),
        )
        if stale_ids:
            await index.delete(ids=stale_ids, namespace=export_config_id)

        vectors = [
            {
                "id": _point_id(o),
                "values": embeddings[i],
                "metadata": _build_metadata(o),
            }
            for i, o in enumerate(outputs)
        ]
        await index.upsert(vectors=vectors, namespace=export_config_id)

    async def delete_file(
        self,
        *,
        parsed_directory_file_id: str,
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> int:
        index, _ = await self._target_service.ensure_index(
            self._target_id, embedding_model
        )
        ids = await self._list_ids_with_prefix(
            index=index,
            namespace=export_config_id,
            prefix=_file_id_prefix(parsed_directory_file_id),
        )
        if not ids:
            return 0
        await index.delete(ids=ids, namespace=export_config_id)
        return len(ids)

    async def delete_files(
        self,
        *,
        parsed_directory_file_ids: list[str],
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> int:
        if not parsed_directory_file_ids:
            return 0
        index, _ = await self._target_service.ensure_index(
            self._target_id, embedding_model
        )

        # No batched cross-file prefix listing in Pinecone, so we enumerate
        # per file. Each call is cheap (one HTTP round trip per page), and
        # the typical batch size in this codebase is small.
        total = 0
        for file_id in parsed_directory_file_ids:
            ids = await self._list_ids_with_prefix(
                index=index,
                namespace=export_config_id,
                prefix=_file_id_prefix(file_id),
            )
            if not ids:
                continue
            await index.delete(ids=ids, namespace=export_config_id)
            total += len(ids)
        return total

    async def list_snapshots(
        self,
        *,
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> list[ExportSnapshot]:
        """Return one snapshot per file currently in the namespace.

        Pinecone's ``list_paginated`` returns ids only, so we:

        1. enumerate every id in the namespace (no prefix),
        2. group ids by file (the substring before ``#``), keeping a
           single representative id per file — chunk #0 when present,
           otherwise the first id seen,
        3. ``fetch`` those representatives in batches to read metadata.

        Each file's snapshot fields (metadata hash, content fingerprint,
        timestamps) are written uniformly across all its chunks, so any
        single chunk faithfully represents the file.
        """
        index, _ = await self._target_service.ensure_index(
            self._target_id, embedding_model
        )

        representative_id_by_file: dict[str, str] = {}
        pagination_token: str | None = None

        for _ in range(_LIST_MAX_PAGES):
            response = await index.list_paginated(
                limit=_LIST_PAGE_LIMIT,
                pagination_token=pagination_token,
                namespace=export_config_id,
            )
            vectors = response.vectors
            for v in vectors:
                vid = v.id
                if not isinstance(vid, str):
                    continue
                file_id, chunk = _split_id(vid)
                if chunk is None:
                    continue
                existing = representative_id_by_file.get(file_id)
                # Prefer chunk #0 as the canonical representative because
                # ``export()`` always emits it. Otherwise keep the first id
                # we saw.
                if existing is None or chunk == "0":
                    representative_id_by_file[file_id] = vid

            pagination = response.pagination
            pagination_token = pagination.next if pagination is not None else None
            if not pagination_token:
                break

        if not representative_id_by_file:
            return []

        # Fetch representatives in batches to populate snapshot fields.
        all_ids = list(representative_id_by_file.values())
        snapshots: list[ExportSnapshot] = []
        for start in range(0, len(all_ids), _FETCH_BATCH_SIZE):
            chunk_ids = all_ids[start : start + _FETCH_BATCH_SIZE]
            fetch_response = await index.fetch(
                ids=chunk_ids, namespace=export_config_id
            )
            # Pinecone's FetchResponse exposes vectors as a mapping id -> Vector.
            vectors = fetch_response.vectors
            for vid, vec in vectors.items():
                md = vec.metadata
                if not isinstance(md, dict):
                    md = {}
                file_id = md.get(PARSED_DIRECTORY_FILE_ID_FIELD) or _split_id(vid)[0]
                if not isinstance(file_id, str):
                    continue
                snapshots.append(
                    ExportSnapshot(
                        parsed_directory_file_id=file_id,
                        metadata_fields_hash=_as_str(
                            md.get(METADATA_FIELDS_HASH_FIELD)
                        ),
                        content_fingerprint_hash=_as_str(
                            md.get("content_fingerprint_hash")
                        ),
                        source_modified_at=_parse_datetime(
                            md.get("source_modified_at")
                        ),
                        exported_at=_parse_datetime(md.get("exported_at")),
                    )
                )

        return snapshots

    async def _list_ids_with_prefix(
        self,
        *,
        index: AsyncIndex,
        namespace: str,
        prefix: str,
    ) -> list[str]:
        """Page through every id in ``namespace`` matching ``prefix``."""
        ids: list[str] = []
        pagination_token: str | None = None
        for _ in range(_LIST_MAX_PAGES):
            response = await index.list_paginated(
                prefix=prefix,
                limit=_LIST_PAGE_LIMIT,
                pagination_token=pagination_token,
                namespace=namespace,
            )
            vectors = response.vectors
            for v in vectors:
                vid = v.id
                if isinstance(vid, str):
                    ids.append(vid)
            pagination = response.pagination
            pagination_token = pagination.next if pagination is not None else None
            if not pagination_token:
                break
        return ids


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


pinecone_exporter = PineconeExporter()


@functools.lru_cache(maxsize=1)
def get_pinecone_exporter() -> PineconeExporter:
    return pinecone_exporter
