"""Qdrant export target implementation.

Writes :class:`ExportedOutput` chunks to a Qdrant collection, one
collection per export config (the recommended multi-tenant shape: cheap
to create and fully isolated for filter/delete operations).

Uses a delete-then-upsert strategy per file so re-parsed files (which
may produce a different number of chunks) are fully replaced rather
than duplicated.
"""

import functools
import uuid
from typing import Any

from qdrant_client.http import models as qmodels

from ..embeddings import SupportedEmbeddingModel
from ..schema import ExportedOutput
from ..services.qdrant import (
    DEFAULT_TARGET_ID,
    EXPORT_CONFIG_ID_FIELD,
    METADATA_FIELDS_HASH_FIELD,
    PARSED_DIRECTORY_FILE_ID_FIELD,
    TENANT_ID_FIELD,
    QdrantService,
    get_qdrant_service,
)
from ..utils import compute_metadata_hash
from .base import Exporter
from .snapshot import ExportSnapshot

# Stable UUID namespace used to derive deterministic Qdrant point ids from
# ``(parsed_directory_file_id, chunk_index)`` strings. Qdrant requires
# point ids to be either unsigned ints or UUIDs; uuid5 with a fixed
# namespace gives us deterministic, collision-free ids per chunk so
# delete-then-upsert is idempotent across re-exports.
_POINT_ID_NAMESPACE = uuid.UUID("4f3c8e8e-6b53-4a51-9c0e-1d0c8e3a4f4a")

# Snapshot pagination caps: per-page chunk fetch + max number of pages.
# Qdrant ``scroll`` returns points sorted by id ascending and the chunks of
# a single file aren't guaranteed to be contiguous (uuid5 over file_id +
# chunk_index doesn't preserve any grouping), so the dedupe loop has to
# fan over all chunks matching the export config. Combined ceiling =
# _SNAPSHOT_MAX_PAGES * _SNAPSHOT_PAGE_SIZE chunks scanned per call.
_SNAPSHOT_PAGE_SIZE = 1000
_SNAPSHOT_MAX_PAGES = 100


def _point_id(output: ExportedOutput) -> str:
    """Deterministic UUID5 point id for a chunk.

    Combines ``parsed_directory_file_id`` and ``chunk_index`` so repeated
    exports of the same chunk land on the same point id (idempotent
    upsert). ``export_config_id`` is intentionally omitted because each
    export config maps to its own collection — see ``ensure_collection``.
    """
    return str(
        uuid.uuid5(
            _POINT_ID_NAMESPACE,
            f"{output.parsed_directory_file_id}:{output.chunk_index}",
        )
    )


def _build_payload(output: ExportedOutput) -> dict[str, Any]:
    """Project an ExportedOutput into a Qdrant payload.

    Stores the model dump (sans embedding, which lives on the point's
    ``vector``) plus a pre-computed metadata hash so
    :meth:`list_snapshots` doesn't have to re-hash a JSON-round-tripped
    dict (see the equivalent reasoning in the turbopuffer exporter).

    Raises:
        ValueError: if ``tenant_id`` or ``export_config_id`` is empty on
            the row. Snapshots and deletes are scoped by these two
            fields, so a row written without them would be unreachable
            for cleanup. Better to fail at write time than silently leak
            orphaned points.
    """
    payload = output.model_dump(mode="json")
    if not payload.get(TENANT_ID_FIELD) or not payload.get(EXPORT_CONFIG_ID_FIELD):
        raise ValueError(
            "ExportedOutput is missing tenant_id and/or export_config_id; "
            "Qdrant snapshots and deletes filter on both. Refusing to write a "
            "row that would be unreachable."
        )

    if output.metadata_fields:
        meta_hash = compute_metadata_hash(output.metadata_fields)
        if meta_hash is not None:
            payload[METADATA_FIELDS_HASH_FIELD] = meta_hash
    return payload


def _file_filter(
    *, export_config_id: str, parsed_directory_file_id: str
) -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=EXPORT_CONFIG_ID_FIELD,
                match=qmodels.MatchValue(value=export_config_id),
            ),
            qmodels.FieldCondition(
                key=PARSED_DIRECTORY_FILE_ID_FIELD,
                match=qmodels.MatchValue(value=parsed_directory_file_id),
            ),
        ]
    )


def _files_filter(
    *, export_config_id: str, parsed_directory_file_ids: list[str]
) -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=EXPORT_CONFIG_ID_FIELD,
                match=qmodels.MatchValue(value=export_config_id),
            ),
            qmodels.FieldCondition(
                key=PARSED_DIRECTORY_FILE_ID_FIELD,
                match=qmodels.MatchAny(any=parsed_directory_file_ids),
            ),
        ]
    )


def _config_filter(export_config_id: str) -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=EXPORT_CONFIG_ID_FIELD,
                match=qmodels.MatchValue(value=export_config_id),
            ),
        ]
    )


class QdrantExporter(Exporter):
    """Writes ExportedOutput chunks to a Qdrant collection."""

    def __init__(
        self,
        *,
        target_id: str = DEFAULT_TARGET_ID,
        target_service: QdrantService | None = None,
    ) -> None:
        self._target_service = target_service or get_qdrant_service()
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
            raise ValueError("`embeddings` must be provided for Qdrant export")
        if not outputs:
            return

        client, collection = await self._target_service.ensure_collection(
            self._target_id, embedding_model
        )

        # All outputs in a single export() call belong to the same file+config.
        parsed_directory_file_id = outputs[0].parsed_directory_file_id
        export_config_id = outputs[0].export_config_id

        # Replace previous chunks for this (config, file) pair, then upsert
        # the new ones. Qdrant doesn't expose a "delete + upsert" atomic
        # primitive, so we issue them as two separate calls. The deterministic
        # uuid5 ids make the upsert side idempotent on its own; the explicit
        # delete handles the case where the re-parse produced fewer chunks.
        await client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(
                filter=_file_filter(
                    export_config_id=export_config_id,
                    parsed_directory_file_id=parsed_directory_file_id,
                )
            ),
            wait=True,
        )

        points = [
            qmodels.PointStruct(
                id=_point_id(o),
                vector=embeddings[i],
                payload=_build_payload(o),
            )
            for i, o in enumerate(outputs)
        ]
        await client.upsert(collection_name=collection, points=points, wait=True)

    async def delete_file(
        self,
        *,
        parsed_directory_file_id: str,
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> int:
        client, collection = await self._target_service.ensure_collection(
            self._target_id, embedding_model
        )
        flt = _file_filter(
            export_config_id=export_config_id,
            parsed_directory_file_id=parsed_directory_file_id,
        )
        # Qdrant's delete result doesn't include a row count; count first so
        # the protocol's return value carries a real number and the caller
        # can log / alert on unexpectedly-large cleanups.
        count = (await client.count(collection_name=collection, count_filter=flt)).count
        await client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(filter=flt),
            wait=True,
        )
        return count

    async def delete_files(
        self,
        *,
        parsed_directory_file_ids: list[str],
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> int:
        if not parsed_directory_file_ids:
            return 0
        client, collection = await self._target_service.ensure_collection(
            self._target_id, embedding_model
        )
        flt = _files_filter(
            export_config_id=export_config_id,
            parsed_directory_file_ids=parsed_directory_file_ids,
        )
        count = (await client.count(collection_name=collection, count_filter=flt)).count
        await client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(filter=flt),
            wait=True,
        )
        return count

    async def list_snapshots(
        self,
        *,
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> list[ExportSnapshot]:
        """Return one snapshot per file currently in the collection.

        Pages over all points matching ``export_config_id`` and dedupes by
        ``parsed_directory_file_id``. The first row seen per file wins;
        because the per-file payload values that snapshots care about
        (metadata hash, content fingerprint, timestamps) are written
        uniformly across all chunks of a file, any chunk is a faithful
        representative.
        """
        client, collection = await self._target_service.ensure_collection(
            self._target_id, embedding_model
        )

        seen: dict[str, ExportSnapshot] = {}
        offset: Any = None
        flt = _config_filter(export_config_id)

        for _ in range(_SNAPSHOT_MAX_PAGES):
            records, offset = await client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=_SNAPSHOT_PAGE_SIZE,
                offset=offset,
                with_payload=[
                    PARSED_DIRECTORY_FILE_ID_FIELD,
                    METADATA_FIELDS_HASH_FIELD,
                    "content_fingerprint_hash",
                    "source_modified_at",
                    "exported_at",
                ],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                file_id = payload.get(PARSED_DIRECTORY_FILE_ID_FIELD)
                if not isinstance(file_id, str) or file_id in seen:
                    continue
                seen[file_id] = ExportSnapshot(
                    parsed_directory_file_id=file_id,
                    metadata_fields_hash=_as_str(
                        payload.get(METADATA_FIELDS_HASH_FIELD)
                    ),
                    content_fingerprint_hash=_as_str(
                        payload.get("content_fingerprint_hash")
                    ),
                    source_modified_at=_as_str(payload.get("source_modified_at")),  # type: ignore[arg-type]
                    exported_at=_as_str(payload.get("exported_at")),  # type: ignore[arg-type]
                )
            if offset is None:
                break

        return list(seen.values())


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


qdrant_exporter = QdrantExporter()


@functools.lru_cache(maxsize=1)
def get_qdrant_exporter() -> QdrantExporter:
    return qdrant_exporter
