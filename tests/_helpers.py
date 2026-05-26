"""Shared test helpers for the export-target test suites."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from index_v2_data_sinks.schema import ExportedOutputV1


def make_exported(
    *,
    file_id: str = "file-1",
    config_id: str = "cfg-1",
    chunk_index: int = 0,
    metadata: dict[str, Any] | None = None,
    tenant_id: str = "proj-1",
) -> ExportedOutputV1:
    """Build an ExportedOutputV1 with sensible defaults for export tests."""
    return ExportedOutputV1(
        version="v1",
        content=f"chunk-{chunk_index}",
        chunk_index=chunk_index,
        chunk_start_char=0,
        chunk_end_char=10,
        chunk_token_count=5,
        parsed_directory_file_id=file_id,
        page_range_start=1,
        page_range_end=1,
        tenant_id=tenant_id,
        export_config_id=config_id,
        metadata_fields=metadata,
        exported_at=datetime.now(UTC),
    )


class AsyncIter:
    """Minimal async iterator over a finite list of items."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> "AsyncIter":
        self._iter = iter(self._items)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration as e:
            raise StopAsyncIteration from e
