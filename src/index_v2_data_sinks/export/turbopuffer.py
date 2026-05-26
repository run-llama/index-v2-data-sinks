"""Turbopuffer export target implementation.

Writes :class:`ExportedOutput` chunks to a Turbopuffer namespace, one
namespace per export config (the recommended multi-tenant shape: cheap
to create, isolated, and idle namespaces are essentially free thanks to
tpuf's object-storage-first architecture).

Uses a delete-then-upsert strategy per file so re-parsed files (which
may produce a different number of chunks) are fully replaced rather
than duplicated.
"""

import functools
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from ..embeddings import SupportedEmbeddingModel
from ..schema import (
    ExportedOutput,
    MetadataDict,
    MetadataIndexingMode,
    MetadataIndexingModeValues,
    MetadataValue,
)
from ..services.turbopuffer import (
    EXPORT_CONFIG_ID_FIELD,
    METADATA_FIELDS_HASH_FIELD,
    TENANT_ID_FIELD,
    SnapshotEnumerationCapExceeded,
    TurbopufferService,
    filter_and,
    filter_eq,
    filter_gt,
    filter_in,
    get_turbopuffer_service,
    metadata_attr_name,
    translate_tpuf_errors,
)
from ..utils import compute_metadata_hash
from .base import Exporter
from .snapshot import ExportSnapshot

if TYPE_CHECKING:
    from turbopuffer.types import Row
    from turbopuffer.types.attribute_schema_param import (
        AttributeSchemaParam,
    )
    from turbopuffer.types.custom import Filter
    from turbopuffer.types.row_param import RowParam


# Snapshot pagination caps: per-page chunk fetch + max number of pages.
# Combined ceiling = _SNAPSHOT_MAX_PAGES * _SNAPSHOT_PAGE_SIZE chunks scanned
# per ``list_snapshots`` call. Bumping the ceiling is fine in the short term;
# at extreme scales a server-side group_by shape is the right answer instead.
_SNAPSHOT_PAGE_SIZE = 1000
_SNAPSHOT_MAX_PAGES = 100


def _row_id(output: ExportedOutput) -> str:
    """Stable per-chunk id used as the tpuf row id.

    ``export_config_id`` is intentionally omitted: each export config maps to
    its own tpuf namespace (see ``get_namespace(self._target_id)``), so rows
    in different namespaces never collide on id. Dropping the redundant
    prefix also keeps the id under tpuf's 64-byte cap — with prefixed
    ``BaseWithStringId`` values the old shape was 67 bytes and tpuf rejected
    the write.
    """
    return f"{output.parsed_directory_file_id}:{output.chunk_index}"


def _coerce_metadata_value(
    value: MetadataValue, mode: MetadataIndexingMode
) -> "str | int | float | bool | None":
    """Coerce a metadata value into the type the namespace will record.

    In ``stringify`` mode every non-None value becomes a string. This is
    what makes per-key attribute types stable across heterogeneous
    documents: a key that arrives as ``1`` in one document and ``1.5`` in
    the next still lands as ``str`` both times, so the namespace schema
    set on first write keeps validating later writes. The
    JSON-round-tripped ``metadata_fields`` blob still carries the
    original typed value for retrieval-time reconstruction.

    In ``infer`` mode the value is passed through unchanged so range
    filtering on numeric/bool attributes keeps working. The caller is
    then responsible for keeping per-key types stable across the export
    config's lifetime; conflicting types in later writes will be rejected
    by turbopuffer's schema validation.

    List metadata is preserved in the JSON-round-tripped ``metadata_fields``
    blob, but is not promoted to ``mf__*`` filter attributes in this slice.
    """
    if value is None or isinstance(value, list):
        return None
    if mode == MetadataIndexingModeValues.STRINGIFY:
        # ``str(True)`` is ``"True"`` which round-trips cleanly enough for
        # equality filters; if a caller needs literal "true"/"false" they
        # should normalize upstream.
        return str(value)
    return value


def _to_attributes(
    output: ExportedOutput, *, metadata_indexing_mode: MetadataIndexingMode
) -> dict[str, Any]:
    """Project an ExportedOutput into tpuf attributes, omitting the embedding.

    ``metadata_fields`` gets three encodings, each serving a distinct read
    path:

    * Stored as a JSON string under ``metadata_fields`` so retrieval can
      reconstruct ``RetrievalResult.metadata`` for callers. This blob
      preserves the *original* metadata values regardless of indexing
      mode, so retrieval results stay typed even when the indexed
      ``mf__*`` attributes have been stringified.
    * Scalar keys are mirrored to top-level ``mf__<key>`` attributes so
      Mongo-parity filters land on tpuf's indexed fast path. List metadata
      stays only in ``metadata_fields`` until list-filter semantics are
      explicitly supported. The promoted scalar value is shaped by
      ``metadata_indexing_mode``.
    * The hash of the original dict is computed at write time and stored
      as ``metadata_fields_hash`` so the snapshot diff doesn't have to
      JSON-decode and re-hash the round-tripped value. Re-hashing the
      decoded dict would be a hash-stability liability if the metadata
      schema ever widens to include non-JSON-stable types (datetime,
      nested dict, etc.); pre-computing keeps the hash on the
      source-side dict and dodges that trap entirely.

    Raises:
        ValueError: if ``tenant_id`` or ``export_config_id`` is empty on the
            row. Retrieval filters are scoped by these two attributes, so a
            row that lands in tpuf without them is unreachable. Failing
            loudly at write time is much better than silently-empty
            ``/retrieve`` responses for that row's chunks.
    """
    raw = output.model_dump(mode="json", exclude={"attachments", "metadata_fields"})
    attrs: dict[str, Any] = {k: v for k, v in raw.items() if v is not None}

    if not attrs.get(TENANT_ID_FIELD) or not attrs.get(EXPORT_CONFIG_ID_FIELD):
        raise ValueError(
            "ExportedOutput is missing tenant_id and/or export_config_id; "
            "tpuf retrieval filters require both. Refusing to write a row "
            "that would be unreachable."
        )

    if output.metadata_fields:
        attrs["metadata_fields"] = json.dumps(output.metadata_fields, sort_keys=True)
        # Pre-compute the hash on the source-side dict so the snapshot
        # diff side just reads an attribute instead of re-hashing the
        # JSON-decoded value.
        meta_hash = compute_metadata_hash(output.metadata_fields)
        if meta_hash is not None:
            attrs[METADATA_FIELDS_HASH_FIELD] = meta_hash
        for key, value in output.metadata_fields.items():
            coerced = _coerce_metadata_value(value, metadata_indexing_mode)
            if coerced is None:
                continue
            attrs[metadata_attr_name(key)] = coerced
    return attrs


# Schema applied on every write. tpuf treats this as idempotent — the first
# write against a fresh namespace registers it; subsequent calls are no-ops.
# ``content.full_text_search`` is required for the BM25 path used by
# TurbopufferRetrievalService; the rest are filterable attributes.
_NAMESPACE_SCHEMA: "dict[str, AttributeSchemaParam]" = {
    "content": {"type": "string", "full_text_search": True},
    TENANT_ID_FIELD: {"type": "string", "filterable": True},
    EXPORT_CONFIG_ID_FIELD: {"type": "string", "filterable": True},
    "parsed_directory_file_id": {"type": "string", "filterable": True},
    "source_file_id": {"type": "string", "filterable": True},
    "content_fingerprint_hash": {"type": "string", "filterable": True},
    "page_range_start": {"type": "uint", "filterable": True},
    "page_range_end": {"type": "uint", "filterable": True},
    "chunk_index": {"type": "uint", "filterable": True},
    # Declared explicitly so retrieval's ``include_attributes`` resolves
    # even when the first batch's rows happen to have these as ``None``
    # (filtered out of ``attrs`` and never seen by tpuf's inference).
    "chunk_start_char": {"type": "uint", "filterable": False},
    "chunk_end_char": {"type": "uint", "filterable": False},
    "chunk_token_count": {"type": "uint", "filterable": False},
    # ISO-8601 strings via ``model_dump(mode="json")``. Both are read by
    # ``list_snapshots`` via ``include_attributes`` and can be ``None`` on
    # write (source_modified_at especially), so declare them explicitly.
    "source_modified_at": {"type": "string", "filterable": False},
    "exported_at": {"type": "string", "filterable": False},
    # JSON-serialized snapshot of the original metadata_fields dict, kept so
    # retrieval results and snapshot diffs can recover the original shape.
    # Filtering goes through the per-key ``mf__<name>`` attributes — see
    # ``_metadata_fields_schema``.
    "metadata_fields": {"type": "string", "filterable": False},
    METADATA_FIELDS_HASH_FIELD: {"type": "string", "filterable": False},
}


def _metadata_fields_schema(
    metadata_fields: MetadataDict,
    *,
    metadata_indexing_mode: MetadataIndexingMode,
) -> "dict[str, AttributeSchemaParam]":
    """Build per-batch schema entries for the flattened ``mf__<key>`` attributes.

    Tpuf infers attribute types on first write, but doesn't guarantee the
    inferred attribute is filterable. Declaring each ``mf__<key>`` explicitly
    with ``filterable: True`` keeps custom-metadata filters on the indexed
    fast path.

    In ``stringify`` mode every ``mf__<key>`` is declared ``string`` so the
    namespace schema is stable across documents with heterogeneous metadata
    types and across batches that see different first-non-None values.

    In ``infer`` mode the type is read off the first non-None scalar value
    in the batch (the same lookup ``_to_attributes`` does, so emitted rows
    match the declared schema). List values are preserved in ``metadata_fields``
    but not promoted in this slice. Mixed scalar types across writes will
    still fail tpuf's schema validation; that's the trade-off opt-in callers
    accept in exchange for typed filters.
    """
    schema: dict[str, AttributeSchemaParam] = {}
    for key, value in metadata_fields.items():
        if value is None:
            continue
        tpuf_type: str
        if metadata_indexing_mode == MetadataIndexingModeValues.STRINGIFY:
            tpuf_type = "string"
        else:
            # bool must be checked before int — bool is a subclass of int.
            if isinstance(value, bool):
                tpuf_type = "bool"
            elif isinstance(value, int):
                tpuf_type = "int"
            elif isinstance(value, float):
                tpuf_type = "float"
            elif isinstance(value, str):
                tpuf_type = "string"
            else:
                continue
        schema[metadata_attr_name(key)] = {"type": tpuf_type, "filterable": True}
    return schema


def _build_file_filter(
    *, export_config_id: str, parsed_directory_file_id: str
) -> "Filter":
    return filter_and(
        [
            filter_eq("export_config_id", export_config_id),
            filter_eq("parsed_directory_file_id", parsed_directory_file_id),
        ]
    )


def _build_files_filter(
    *, export_config_id: str, parsed_directory_file_ids: list[str]
) -> "Filter":
    return filter_and(
        [
            filter_eq("export_config_id", export_config_id),
            filter_in("parsed_directory_file_id", parsed_directory_file_ids),
        ]
    )


class TurbopufferExporter(Exporter):
    """Writes ExportedOutput chunks to a Turbopuffer namespace."""

    def __init__(
        self,
        *,
        target_id: str,
        target_service: TurbopufferService | None = None,
        metadata_indexing_mode: MetadataIndexingMode = MetadataIndexingModeValues.INFER,
    ) -> None:
        # ``target_id`` is required (keyword-only) because it determines the
        # tpuf namespace name. Defaulting to a constant would silently funnel
        # every caller into a shared namespace and collapse tenant isolation
        # — see the factory-bug fix in the original PR.
        if not target_id:
            raise ValueError(
                "TurbopufferExportTarget requires a non-empty target_id; "
                "callers should pass the export config id."
            )
        self._target_service = target_service or get_turbopuffer_service()
        self._target_id = target_id
        self._metadata_indexing_mode = metadata_indexing_mode

    async def export(
        self,
        outputs: list[ExportedOutput],
        *,
        project_id: str,
        embeddings: list[list[float]] | None = None,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> None:
        if embeddings is None:
            raise ValueError("`embeddings` must be provided for Turbopuffer export")
        if not outputs:
            return

        ns = self._target_service.get_namespace(self._target_id)

        # All outputs in a single export() call belong to the same file+config.
        parsed_directory_file_id = outputs[0].parsed_directory_file_id
        export_config_id = outputs[0].export_config_id

        # Replace previous chunks for this (config, file) pair, then write
        # the new ones. Two write() calls rather than one because tpuf
        # applies delete-by-filter and upsert in a single request as a
        # combined operation only when both shapes share a target set,
        # which isn't the case here.
        async with translate_tpuf_errors():
            await ns.write(
                delete_by_filter=_build_file_filter(
                    export_config_id=export_config_id,
                    parsed_directory_file_id=parsed_directory_file_id,
                ),
            )

        rows: list[dict[str, Any]] = [
            {
                "id": _row_id(o),
                "vector": embeddings[i],
                **_to_attributes(
                    o,
                    metadata_indexing_mode=self._metadata_indexing_mode,  # pyright: ignore[reportArgumentType]
                ),
            }
            for i, o in enumerate(outputs)
        ]

        # Extend the base schema with explicit ``mf__<key>`` declarations for
        # every metadata key in this batch, so custom-metadata filters land
        # on the indexed fast path rather than relying on tpuf's auto-infer
        # to make them filterable.
        schema: dict[str, AttributeSchemaParam] = dict(_NAMESPACE_SCHEMA)
        for o in outputs:
            if o.metadata_fields:
                batch_schema = _metadata_fields_schema(
                    o.metadata_fields,
                    metadata_indexing_mode=self._metadata_indexing_mode,  # pyright: ignore[reportArgumentType]
                )
                for attr_name, attr_schema in batch_schema.items():
                    schema.setdefault(attr_name, attr_schema)

        # RowParam is a TypedDict with extra_items; ty can't infer that our
        # dict literals match its shape, hence the cast.
        async with translate_tpuf_errors():
            await ns.write(
                upsert_rows=cast("list[RowParam]", rows),
                distance_metric="cosine_distance",
                schema=schema,
            )

    async def delete_file(
        self,
        *,
        parsed_directory_file_id: str,
        export_config_id: str,
    ) -> int:
        ns = self._target_service.get_namespace(self._target_id)
        async with translate_tpuf_errors():
            result = await ns.write(
                delete_by_filter=_build_file_filter(
                    export_config_id=export_config_id,
                    parsed_directory_file_id=parsed_directory_file_id,
                ),
            )
        return result.rows_affected

    async def delete_files(
        self,
        *,
        parsed_directory_file_ids: list[str],
        export_config_id: str,
    ) -> int:
        if not parsed_directory_file_ids:
            return 0
        ns = self._target_service.get_namespace(self._target_id)
        async with translate_tpuf_errors():
            result = await ns.write(
                delete_by_filter=_build_files_filter(
                    export_config_id=export_config_id,
                    parsed_directory_file_ids=parsed_directory_file_ids,
                ),
            )
        return result.rows_affected

    async def delete_namespace(self) -> None:
        """Drop the entire namespace backing this export target.

        Idempotent — tpuf treats deleting a missing namespace as a no-op.
        """
        ns = self._target_service.get_namespace(self._target_id)
        async with translate_tpuf_errors():
            await ns.delete_all()

    async def list_snapshots(
        self,
        *,
        export_config_id: str,
    ) -> list[ExportSnapshot]:
        """Return one snapshot per file currently in the tpuf namespace.

        Tpuf has no server-side ``GROUP BY``, so we page through chunks
        ranked by ``parsed_directory_file_id`` ascending and advance the
        cursor with a strict-greater-than filter after each page. Because
        chunks of the same file sort contiguously, the cursor skips every
        already-seen file's remaining chunks rather than re-scanning them.

        Cost: O(distinct_files / files_per_page) round-trips, where
        files_per_page = page_size / avg_chunks_per_file. The hard cap of
        ``_SNAPSHOT_MAX_PAGES`` × ``_SNAPSHOT_PAGE_SIZE`` chunks is a safety
        rail; if a single export config grows beyond that we should swap
        to a ``group_by`` shape rather than just bumping the cap.
        """
        ns = self._target_service.get_namespace(self._target_id)
        seen: dict[str, ExportSnapshot] = {}
        last_seen_id: str | None = None
        cap_exhausted = True

        for _ in range(_SNAPSHOT_MAX_PAGES):
            base_clause = filter_eq("export_config_id", export_config_id)
            page_filter: Filter
            if last_seen_id is None:
                page_filter = base_clause
            else:
                page_filter = filter_and(
                    [
                        base_clause,
                        filter_gt("parsed_directory_file_id", last_seen_id),
                    ]
                )

            async with translate_tpuf_errors():
                page = await ns.query(
                    rank_by=("parsed_directory_file_id", "asc"),
                    filters=page_filter,
                    include_attributes=[
                        "parsed_directory_file_id",
                        METADATA_FIELDS_HASH_FIELD,
                        "content_fingerprint_hash",
                        "source_modified_at",
                        "exported_at",
                    ],
                    top_k=_SNAPSHOT_PAGE_SIZE,
                )
            rows = page.rows or []
            if not rows:
                cap_exhausted = False
                break

            for row in rows:
                file_id = _row_attr(row, "parsed_directory_file_id")
                if not isinstance(file_id, str) or file_id in seen:
                    continue
                # Read the pre-computed hash stored at write time. No
                # JSON decode + re-hash on this path: see _to_attributes
                # for the reasoning.
                seen[file_id] = ExportSnapshot(
                    parsed_directory_file_id=file_id,
                    metadata_fields_hash=_row_attr_str(row, METADATA_FIELDS_HASH_FIELD),
                    content_fingerprint_hash=_row_attr_str(
                        row, "content_fingerprint_hash"
                    ),
                    source_modified_at=_parse_dt(_row_attr(row, "source_modified_at")),
                    exported_at=_parse_dt(_row_attr(row, "exported_at")),
                )

            tail_id = _row_attr(rows[-1], "parsed_directory_file_id")
            if not isinstance(tail_id, str):
                cap_exhausted = False
                break
            last_seen_id = tail_id
            # Short-circuit when tpuf gave us a partial page; no more rows.
            if len(rows) < _SNAPSHOT_PAGE_SIZE:
                cap_exhausted = False
                break

        if cap_exhausted:
            # Hit the page-count ceiling without exhausting rows. Returning
            # an incomplete snapshot would make the diff service treat the
            # missing files as needing re-export and repeat that work
            # forever. Fail loudly instead — the caller (alerting + manual
            # reindex, or moving the config off the tpuf path) is better
            # than silent corruption.
            raise SnapshotEnumerationCapExceeded(
                f"list_snapshots for export_config_id={export_config_id!r} hit the "
                f"{_SNAPSHOT_MAX_PAGES * _SNAPSHOT_PAGE_SIZE}-chunk cap before "
                "exhausting rows. Snapshot is incomplete; the diff service must "
                "not act on it. Time to switch list_snapshots to a server-side "
                "group_by shape."
            )

        return list(seen.values())


def _row_attr(row: "Row", name: str) -> object:
    """Read an extra attribute from a tpuf Row.

    ``Row`` exposes ``id`` and ``vector`` as fields; other attributes
    requested via ``include_attributes`` arrive as ``__pydantic_extra__``.
    Use ``getattr`` with a default so callers can branch on absence
    without raising.
    """
    return getattr(row, name, None)


def _row_attr_str(row: "Row", name: str) -> str | None:
    value = _row_attr(row, name)
    return value if isinstance(value, str) else None


def _parse_dt(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


@functools.lru_cache(maxsize=1)
def get_turbopuffer_exporter() -> TurbopufferExporter:
    target = os.getenv("TURBOPUFFER_TARGET_ID", "DEFAULT")
    tpuf_exporter = TurbopufferExporter(target_id=target)
    return tpuf_exporter
