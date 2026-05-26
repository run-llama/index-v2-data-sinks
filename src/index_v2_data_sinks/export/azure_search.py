"""Azure AI Search export target implementation.

Writes :class:`ExportedOutput` chunks to an Azure AI Search index.
Multi-tenancy shape:

- one **index** per (target_id, embedding_model) — required because
  Azure AI Search indexes are schema-locked, including vector
  dimension;
- all export configs share an index and are isolated via an
  ``export_config_id`` field + OData filter. Azure AI Search has no
  first-class namespace primitive, so field-scoped filters are how
  per-tenant snapshot/delete work (same shape as Mongo / PGVector /
  Qdrant).

Document keys are deterministic
``"{parsed_directory_file_id}__{chunk_index}"`` strings so re-exports
are idempotent upserts (``merge_or_upload_documents``). The separator
is double-underscore (not ``#``) because Azure AI Search document keys
are restricted to letters, digits, underscores, dashes, and equals
signs — ``#`` is rejected at upload time.

Filtered deletes are not directly supported (Azure only deletes by
document key), so the exporter enumerates keys with a ``search`` call
that selects only the key field plus the change-detection fields, then
deletes by key. The change-detection fields ride along on the same
request so :meth:`list_snapshots` only pays one round-trip per page.
"""

import functools
import json
from datetime import datetime
from typing import Any

from azure.core.exceptions import HttpResponseError
from azure.search.documents.aio import SearchClient

from ..embeddings import SupportedEmbeddingModel
from ..schema import ExportedOutput
from ..services.azure_search import (
    DEFAULT_TARGET_ID,
    EMBEDDING_FIELD,
    EXPORT_CONFIG_ID_FIELD,
    KEY_FIELD,
    METADATA_FIELDS_HASH_FIELD,
    PARSED_DIRECTORY_FILE_ID_FIELD,
    TENANT_ID_FIELD,
    AzureSearchService,
    get_azure_search_service,
)
from ..utils import compute_metadata_hash
from .base import Exporter
from .snapshot import ExportSnapshot

# Separator between file id and chunk index in a document key. Azure
# document keys allow ``[A-Za-z0-9_\-=]`` only, so we avoid ``#`` / ``:``
# and use double-underscore which is unambiguous given file ids are
# typically UUIDs (no underscores).
_KEY_SEP = "__"

# Pagination caps for enumeration paths.
#
# ``search`` accepts ``top`` up to 1000 per call. We page with
# ``$skip`` (also capped at 100000 by Azure) but cap our own page count
# defensively so a runaway index can't stall a snapshot scan.
_SEARCH_PAGE_LIMIT = 1000
_SEARCH_MAX_PAGES = 100

# Azure's batch upload/delete limit is 1000 documents per request.
_UPLOAD_BATCH_SIZE = 1000


def _document_key(output: ExportedOutput) -> str:
    """Deterministic Azure document key for a chunk.

    Uses ``{file_id}{__}{chunk_index}`` so:
    - re-exports of the same chunk overwrite (idempotent merge_or_upload),
    - we can reconstruct ``parsed_directory_file_id`` from the key when
      enumerating, which lets the snapshot loop work without a payload
      round-trip if the filterable field is somehow stripped.
    """
    return f"{output.parsed_directory_file_id}{_KEY_SEP}{output.chunk_index}"


def _build_document(output: ExportedOutput, embedding: list[float]) -> dict[str, Any]:
    """Project an ExportedOutput + embedding into an Azure AI Search document.

    The shape mirrors the field list in
    ``services/azure_search._build_index_definition``. Two design
    points:

    - ``attachments`` and ``metadata_fields`` are JSON-encoded into
      single string fields. Azure supports complex types but they're
      awkward to evolve; the round-trip via ``json.loads`` is cheap on
      the read side and keeps the schema stable.
    - ``metadata_fields_hash`` is precomputed at write time so
      :meth:`list_snapshots` doesn't have to re-hash a JSON-round-tripped
      payload (same reasoning as the Pinecone/Qdrant exporters).

    Raises:
        ValueError: if ``tenant_id`` or ``export_config_id`` is empty
            on the row. Both are required for snapshot/delete scoping;
            a row written without them would be unreachable to cleanup
            paths, and silent data leaks are worse than a loud failure.
    """
    if not output.tenant_id or not output.export_config_id:
        raise ValueError(
            "ExportedOutput is missing tenant_id and/or export_config_id; "
            "Azure AI Search snapshots and deletes filter on both. Refusing "
            "to write a row that would be unreachable."
        )

    doc: dict[str, Any] = {
        KEY_FIELD: _document_key(output),
        TENANT_ID_FIELD: output.tenant_id,
        EXPORT_CONFIG_ID_FIELD: output.export_config_id,
        PARSED_DIRECTORY_FILE_ID_FIELD: output.parsed_directory_file_id,
        "chunk_index": output.chunk_index,
        "page_range_start": output.page_range_start,
        "page_range_end": output.page_range_end,
        # ``content`` is also a top-level retrievable field so callers
        # reading search results don't need a second hop to fetch chunk
        # text. Mirrors the Pinecone/Turbopuffer attribute model.
        "content": output.content,
        EMBEDDING_FIELD: embedding,
    }
    if output.source_file_id is not None:
        doc["source_file_id"] = output.source_file_id
    if output.chunk_start_char is not None:
        doc["chunk_start_char"] = output.chunk_start_char
    if output.chunk_end_char is not None:
        doc["chunk_end_char"] = output.chunk_end_char
    if output.chunk_token_count is not None:
        doc["chunk_token_count"] = output.chunk_token_count
    if output.content_fingerprint_hash is not None:
        doc["content_fingerprint_hash"] = output.content_fingerprint_hash
    if output.source_modified_at is not None:
        # Azure expects ISO 8601 with an offset; ``datetime.isoformat``
        # produces the right shape provided the datetime is
        # timezone-aware (which ``ExportedOutputV1.build`` ensures via
        # ``datetime.now(UTC)``).
        doc["source_modified_at"] = output.source_modified_at.isoformat()
    if output.exported_at is not None:
        doc["exported_at"] = output.exported_at.isoformat()

    if output.metadata_fields:
        meta_hash = compute_metadata_hash(output.metadata_fields)
        if meta_hash is not None:
            doc[METADATA_FIELDS_HASH_FIELD] = meta_hash
        doc["metadata_fields_json"] = json.dumps(output.metadata_fields)

    if output.attachments:
        doc["attachments_json"] = json.dumps(
            [a.model_dump(mode="json") for a in output.attachments]
        )

    return doc


def _escape_odata_string(value: str) -> str:
    """Escape a string literal for safe inclusion in an OData filter.

    Azure's OData dialect uses single quotes around string literals and
    requires single quotes inside to be doubled. We don't expect quotes
    in our id space (file ids are UUID-like, config ids are app-issued
    slugs), but defending against injection here is cheap.
    """
    return value.replace("'", "''")


def _file_filter(*, export_config_id: str, parsed_directory_file_id: str) -> str:
    return (
        f"{EXPORT_CONFIG_ID_FIELD} eq '{_escape_odata_string(export_config_id)}' and "
        f"{PARSED_DIRECTORY_FILE_ID_FIELD} eq "
        f"'{_escape_odata_string(parsed_directory_file_id)}'"
    )


def _files_filter(
    *, export_config_id: str, parsed_directory_file_ids: list[str]
) -> str:
    # ``search.in`` is Azure's idiomatic way to match against a set of
    # values without a long ``or`` chain. The second argument is a
    # delimiter string, kept ASCII and definitely-not-in-ids: comma.
    joined = ",".join(_escape_odata_string(f) for f in parsed_directory_file_ids)
    return (
        f"{EXPORT_CONFIG_ID_FIELD} eq '{_escape_odata_string(export_config_id)}' and "
        f"search.in({PARSED_DIRECTORY_FILE_ID_FIELD}, '{joined}', ',')"
    )


def _config_filter(export_config_id: str) -> str:
    return f"{EXPORT_CONFIG_ID_FIELD} eq '{_escape_odata_string(export_config_id)}'"


def _parse_datetime(value: Any) -> datetime | None:
    """Parse an Azure-returned datetime field.

    The SDK returns ``datetime`` objects directly when the field is
    typed as ``DateTimeOffset``. For robustness against future SDK
    changes (or older index schemas), also accept ISO 8601 strings.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class AzureSearchExporter(Exporter):
    """Writes ExportedOutput chunks to an Azure AI Search index."""

    def __init__(
        self,
        *,
        target_id: str = DEFAULT_TARGET_ID,
        target_service: AzureSearchService | None = None,
    ) -> None:
        self._target_service = target_service or get_azure_search_service()
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
            raise ValueError("`embeddings` must be provided for Azure AI Search export")
        if not outputs:
            return

        client, _ = await self._target_service.ensure_index(
            self._target_id, embedding_model
        )

        # All outputs in a single export() call belong to the same file+config.
        parsed_directory_file_id = outputs[0].parsed_directory_file_id
        export_config_id = outputs[0].export_config_id

        # Replace previous chunks for this (config, file) pair. Azure has
        # no "delete by filter" endpoint, so we enumerate keys via
        # ``search`` and delete by key. The deterministic key scheme
        # means the upsert side is idempotent on its own; the explicit
        # delete handles the case where the re-parse produced fewer
        # chunks (and so leaves orphan keys behind the new write set).
        stale_keys = await self._list_keys_matching(
            client=client,
            filter_expr=_file_filter(
                export_config_id=export_config_id,
                parsed_directory_file_id=parsed_directory_file_id,
            ),
        )
        if stale_keys:
            await self._delete_by_keys(client=client, keys=stale_keys)

        documents = [_build_document(o, embeddings[i]) for i, o in enumerate(outputs)]

        # ``merge_or_upload`` is the idempotent variant: it inserts when
        # absent and replaces all listed fields when present, leaving
        # unlisted fields alone. Since we always include every field on
        # every write, the behavior is effectively "upsert".
        for start in range(0, len(documents), _UPLOAD_BATCH_SIZE):
            batch = documents[start : start + _UPLOAD_BATCH_SIZE]
            await client.merge_or_upload_documents(documents=batch)

    async def delete_file(
        self,
        *,
        parsed_directory_file_id: str,
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> int:
        client, _ = await self._target_service.ensure_index(
            self._target_id, embedding_model
        )
        keys = await self._list_keys_matching(
            client=client,
            filter_expr=_file_filter(
                export_config_id=export_config_id,
                parsed_directory_file_id=parsed_directory_file_id,
            ),
        )
        if not keys:
            return 0
        await self._delete_by_keys(client=client, keys=keys)
        return len(keys)

    async def delete_files(
        self,
        *,
        parsed_directory_file_ids: list[str],
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> int:
        if not parsed_directory_file_ids:
            return 0
        client, _ = await self._target_service.ensure_index(
            self._target_id, embedding_model
        )
        keys = await self._list_keys_matching(
            client=client,
            filter_expr=_files_filter(
                export_config_id=export_config_id,
                parsed_directory_file_ids=parsed_directory_file_ids,
            ),
        )
        if not keys:
            return 0
        await self._delete_by_keys(client=client, keys=keys)
        return len(keys)

    async def list_snapshots(
        self,
        *,
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> list[ExportSnapshot]:
        """Return one snapshot per file currently in the index for ``export_config_id``.

        Implementation note: Azure AI Search doesn't expose a "group by"
        primitive, so we page through every document matching the
        filter and dedupe by ``parsed_directory_file_id`` in memory.
        The first chunk per file wins; because the per-file payload
        values that snapshots care about (metadata hash, content
        fingerprint, timestamps) are written uniformly across all
        chunks of a file, any chunk is a faithful representative.

        The ``search`` is intentionally selective: we project only the
        fields snapshots need (no ``content``, no ``embedding``) and
        sort by key, which lets Azure stream results without a large
        sort cost.
        """
        client, _ = await self._target_service.ensure_index(
            self._target_id, embedding_model
        )

        snapshot_fields = [
            PARSED_DIRECTORY_FILE_ID_FIELD,
            METADATA_FIELDS_HASH_FIELD,
            "content_fingerprint_hash",
            "source_modified_at",
            "exported_at",
        ]

        seen: dict[str, ExportSnapshot] = {}
        async for doc in await self._search_filtered(
            client=client,
            filter_expr=_config_filter(export_config_id),
            select=snapshot_fields,
        ):
            file_id = doc.get(PARSED_DIRECTORY_FILE_ID_FIELD)
            if not isinstance(file_id, str) or file_id in seen:
                continue
            seen[file_id] = ExportSnapshot(
                parsed_directory_file_id=file_id,
                metadata_fields_hash=_as_str(doc.get(METADATA_FIELDS_HASH_FIELD)),
                content_fingerprint_hash=_as_str(doc.get("content_fingerprint_hash")),
                source_modified_at=_parse_datetime(doc.get("source_modified_at")),
                exported_at=_parse_datetime(doc.get("exported_at")),
            )

        return list(seen.values())

    async def _search_filtered(
        self,
        *,
        client: SearchClient,
        filter_expr: str,
        select: list[str],
    ):
        """Return an async iterator over documents matching ``filter_expr``.

        Uses the SDK's built-in paging (``AsyncSearchItemPaged``) which
        transparently handles ``@odata.nextLink`` continuations. Yields
        each hit as a plain ``dict``.
        """
        results = await client.search(
            search_text="*",
            filter=filter_expr,
            select=select,
            # Azure caps ``top`` at 1000 per page; the iterator will
            # follow continuation tokens for us up to the service-side
            # cap (~100k by ``$skip`` semantics). Snapshots and cleanups
            # against a single tenant should stay well below that.
            top=_SEARCH_PAGE_LIMIT,
        )
        return results

    async def _list_keys_matching(
        self, *, client: SearchClient, filter_expr: str
    ) -> list[str]:
        """Enumerate document keys matching ``filter_expr``.

        Selects only the key field so result payloads stay small. The
        caller uses the returned keys to issue a delete batch (Azure
        does not support delete-by-filter; only delete-by-key).
        """
        keys: list[str] = []
        async for doc in await self._search_filtered(
            client=client, filter_expr=filter_expr, select=[KEY_FIELD]
        ):
            key = doc.get(KEY_FIELD)
            if isinstance(key, str):
                keys.append(key)
        return keys

    async def _delete_by_keys(self, *, client: SearchClient, keys: list[str]) -> None:
        """Delete documents by key, batched at Azure's 1000-per-request cap.

        ``delete_documents`` accepts ``{KEY_FIELD: <value>}`` shapes;
        the SDK turns them into proper ``@search.action=delete`` rows.
        Any per-row failures surface via ``IndexingResult.succeeded`` on
        the returned list — we raise a single :class:`HttpResponseError`
        if the whole call fails, but rely on the SDK's behavior for
        partial failures (which would otherwise require a per-row retry
        loop unrelated to the snapshot/diff contract this exporter
        exposes).
        """
        for start in range(0, len(keys), _UPLOAD_BATCH_SIZE):
            batch = [{KEY_FIELD: k} for k in keys[start : start + _UPLOAD_BATCH_SIZE]]
            try:
                await client.delete_documents(documents=batch)
            except HttpResponseError:
                # Surface the original error to the caller — there's no
                # safe partial-progress story for a half-applied delete
                # batch, and the pipeline-level retry layer is the right
                # place to decide what to do.
                raise


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


azure_search_exporter = AzureSearchExporter()


@functools.lru_cache(maxsize=1)
def get_azure_search_exporter() -> AzureSearchExporter:
    return azure_search_exporter
