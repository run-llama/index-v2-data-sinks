"""Tests for index_v2_data_sinks.export.turbopuffer.TurbopufferExporter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from index_v2_data_sinks.export.turbopuffer import (
    TurbopufferExporter,
    _row_id,
)
from index_v2_data_sinks.services.turbopuffer import SnapshotEnumerationCapExceeded

from ._helpers import make_exported


def _make_exporter(*, rows_affected: int = 3):
    ns = MagicMock()
    ns.write = AsyncMock(return_value=SimpleNamespace(rows_affected=rows_affected))
    ns.query = AsyncMock(return_value=SimpleNamespace(rows=[]))
    ns.delete_all = AsyncMock()

    target_service = MagicMock()
    target_service.get_namespace = MagicMock(return_value=ns)

    exporter = TurbopufferExporter(target_id="cfg-1", target_service=target_service)
    return exporter, ns


def test_constructor_requires_non_empty_target_id() -> None:
    with pytest.raises(ValueError, match="target_id"):
        TurbopufferExporter(target_id="", target_service=MagicMock())


async def test_export_requires_embeddings() -> None:
    exporter, _ = _make_exporter()
    with pytest.raises(ValueError, match="embeddings"):
        await exporter.export([make_exported()], project_id="proj-1", embeddings=None)


async def test_export_empty_outputs_is_noop() -> None:
    exporter, ns = _make_exporter()
    await exporter.export([], project_id="proj-1", embeddings=[])
    ns.write.assert_not_awaited()


async def test_export_issues_delete_then_upsert_writes() -> None:
    exporter, ns = _make_exporter()
    outputs = [make_exported(chunk_index=0), make_exported(chunk_index=1)]
    await exporter.export(outputs, project_id="proj-1", embeddings=[[0.1], [0.2]])

    # First write is a delete-by-filter, second is the upsert.
    assert ns.write.await_count == 2
    delete_call = ns.write.await_args_list[0]
    assert "delete_by_filter" in delete_call.kwargs
    assert "upsert_rows" not in delete_call.kwargs

    upsert_call = ns.write.await_args_list[1]
    rows = upsert_call.kwargs["upsert_rows"]
    assert [r["id"] for r in rows] == [_row_id(o) for o in outputs]
    assert [r["vector"] for r in rows] == [[0.1], [0.2]]
    # Schema is attached and includes the routing fields.
    schema = upsert_call.kwargs["schema"]
    assert "tenant_id" in schema
    assert "export_config_id" in schema
    assert upsert_call.kwargs["distance_metric"] == "cosine_distance"


async def test_export_rejects_row_missing_routing_fields() -> None:
    exporter, _ = _make_exporter()
    bad = make_exported(tenant_id="")
    with pytest.raises(ValueError, match="tenant_id"):
        await exporter.export([bad], project_id="proj-1", embeddings=[[0.1]])


async def test_export_promotes_metadata_to_mf_attributes() -> None:
    exporter, ns = _make_exporter()
    outputs = [make_exported(metadata={"author": "alice", "year": 2024})]
    await exporter.export(outputs, project_id="proj-1", embeddings=[[0.1]])

    upsert_call = ns.write.await_args_list[1]
    rows = upsert_call.kwargs["upsert_rows"]
    assert "mf__author" in rows[0]
    assert "mf__year" in rows[0]
    # Schema also declares the promoted attributes.
    schema = upsert_call.kwargs["schema"]
    assert "mf__author" in schema and "mf__year" in schema


async def test_delete_file_returns_rows_affected() -> None:
    exporter, ns = _make_exporter(rows_affected=4)
    n = await exporter.delete_file(
        parsed_directory_file_id="file-1", export_config_id="cfg-1"
    )
    assert n == 4
    ns.write.assert_awaited_once()
    assert "delete_by_filter" in ns.write.call_args.kwargs


async def test_delete_files_empty_list_is_noop() -> None:
    exporter, ns = _make_exporter()
    n = await exporter.delete_files(
        parsed_directory_file_ids=[], export_config_id="cfg-1"
    )
    assert n == 0
    ns.write.assert_not_awaited()


async def test_delete_files_returns_rows_affected() -> None:
    exporter, ns = _make_exporter(rows_affected=9)
    n = await exporter.delete_files(
        parsed_directory_file_ids=["a", "b"], export_config_id="cfg-1"
    )
    assert n == 9


async def test_delete_namespace_calls_delete_all() -> None:
    exporter, ns = _make_exporter()
    await exporter.delete_namespace()
    ns.delete_all.assert_awaited_once()


async def test_list_snapshots_pages_until_partial_then_dedupes() -> None:
    exporter, ns = _make_exporter()

    def _row(file_id: str, **extra) -> SimpleNamespace:
        return SimpleNamespace(parsed_directory_file_id=file_id, **extra)

    # First page: full of rows from file-1 + one row from file-2 (partial -> stop).
    page1 = SimpleNamespace(
        rows=[
            _row(
                "file-1",
                metadata_fields_hash="h1",
                content_fingerprint_hash="fp1",
                source_modified_at=None,
                exported_at=None,
            ),
            _row(
                "file-1",
                metadata_fields_hash="h1",
                content_fingerprint_hash="fp1",
                source_modified_at=None,
                exported_at=None,
            ),
            _row(
                "file-2",
                metadata_fields_hash=None,
                content_fingerprint_hash=None,
                source_modified_at=None,
                exported_at=None,
            ),
        ]
    )
    ns.query = AsyncMock(return_value=page1)

    snapshots = await exporter.list_snapshots(export_config_id="cfg-1")
    by_id = {s.parsed_directory_file_id: s for s in snapshots}
    assert set(by_id) == {"file-1", "file-2"}
    assert by_id["file-1"].metadata_fields_hash == "h1"
    assert by_id["file-2"].metadata_fields_hash is None


async def test_list_snapshots_raises_when_cap_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shrink the caps so the test triggers the overflow path quickly with a
    # single full page that doesn't advance the cursor far enough to finish.
    import index_v2_data_sinks.export.turbopuffer as tpuf_export

    monkeypatch.setattr(tpuf_export, "_SNAPSHOT_PAGE_SIZE", 2)
    monkeypatch.setattr(tpuf_export, "_SNAPSHOT_MAX_PAGES", 1)

    exporter, ns = _make_exporter()
    full_page = SimpleNamespace(
        rows=[
            SimpleNamespace(
                parsed_directory_file_id="file-1",
                metadata_fields_hash=None,
                content_fingerprint_hash=None,
                source_modified_at=None,
                exported_at=None,
            ),
            SimpleNamespace(
                parsed_directory_file_id="file-2",
                metadata_fields_hash=None,
                content_fingerprint_hash=None,
                source_modified_at=None,
                exported_at=None,
            ),
        ]
    )
    ns.query = AsyncMock(return_value=full_page)

    with pytest.raises(SnapshotEnumerationCapExceeded):
        await exporter.list_snapshots(export_config_id="cfg-1")
