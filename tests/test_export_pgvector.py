"""Tests for index_v2_data_sinks.export.pgvector.PGVectorExporter.

The exporter goes through SQLAlchemy's async engine API. Everything below
the ``get_table`` boundary is mocked: ``engine.begin()`` /
``engine.connect()`` return an async-context wrapping a connection whose
``execute()`` records the statements issued, so we can assert on the
delete-then-insert sequencing without standing up a real Postgres.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table

from index_v2_data_sinks.export.pgvector import PGVectorExporter

from ._helpers import make_exported


def _build_fake_table() -> Table:
    """Minimal Table with just the columns the exporter touches in filters/inserts."""
    metadata = MetaData()
    return Table(
        "chunks",
        metadata,
        Column("id", String, primary_key=True),
        Column("parsed_directory_file_id", String),
        Column("export_config_id", String),
        Column("chunk_index", Integer),
        Column("metadata_fields", String),
        Column("content_fingerprint_hash", String),
        Column("source_modified_at", String),
        Column("exported_at", String),
    )


class _AsyncCtx:
    """Async context manager that returns the same connection on every enter."""

    def __init__(self, conn: MagicMock) -> None:
        self._conn = conn

    async def __aenter__(self) -> MagicMock:
        return self._conn

    async def __aexit__(self, *exc: object) -> None:
        return None


def _make_exporter(
    *,
    delete_rowcount: int = 3,
    select_rows: list[tuple] | None = None,
    representative_row: object | None = None,
):
    table = _build_fake_table()
    conn = MagicMock()

    # Default execute() result; individual tests override side_effect when
    # they need different return values across multiple .execute() calls.
    result = MagicMock()
    result.rowcount = delete_rowcount
    result.all = MagicMock(return_value=select_rows or [])
    result.first = MagicMock(return_value=representative_row)
    conn.execute = AsyncMock(return_value=result)

    engine = MagicMock()
    engine.begin = MagicMock(return_value=_AsyncCtx(conn))
    engine.connect = MagicMock(return_value=_AsyncCtx(conn))

    target_service = MagicMock()
    target_service.get_table = AsyncMock(return_value=(engine, table))

    exporter = PGVectorExporter(target_service=target_service, target_id="DEFAULT")
    return exporter, conn, table


async def test_export_requires_embeddings() -> None:
    exporter, _, _ = _make_exporter()
    with pytest.raises(ValueError, match="embeddings"):
        await exporter.export([make_exported()], project_id="proj-1", embeddings=None)


async def test_export_empty_outputs_is_noop() -> None:
    exporter, conn, _ = _make_exporter()
    await exporter.export([], project_id="proj-1", embeddings=[])
    conn.execute.assert_not_awaited()


async def test_export_issues_delete_then_insert() -> None:
    exporter, conn, _ = _make_exporter()
    outputs = [make_exported(chunk_index=0), make_exported(chunk_index=1)]
    await exporter.export(outputs, project_id="proj-1", embeddings=[[0.1], [0.2]])

    # First call = DELETE, second call = INSERT with two row dicts.
    assert conn.execute.await_count == 2
    insert_call = conn.execute.await_args_list[1]
    rows = insert_call.args[1]
    assert isinstance(rows, list) and len(rows) == 2
    # Inserted rows carry the embedding and a generated id.
    assert rows[0]["embedding"] == [0.1]
    assert rows[1]["embedding"] == [0.2]
    assert "id" in rows[0] and "id" in rows[1]
    # ``version`` is stripped from the dump before insertion.
    assert "version" not in rows[0]


async def test_delete_file_returns_rowcount() -> None:
    exporter, conn, _ = _make_exporter(delete_rowcount=4)
    n = await exporter.delete_file(
        parsed_directory_file_id="file-1", export_config_id="cfg-1"
    )
    assert n == 4
    conn.execute.assert_awaited_once()


async def test_delete_files_empty_list_is_noop() -> None:
    exporter, conn, _ = _make_exporter()
    n = await exporter.delete_files(
        parsed_directory_file_ids=[], export_config_id="cfg-1"
    )
    assert n == 0
    conn.execute.assert_not_awaited()


async def test_delete_files_executes_single_delete() -> None:
    exporter, conn, _ = _make_exporter(delete_rowcount=7)
    n = await exporter.delete_files(
        parsed_directory_file_ids=["a", "b"], export_config_id="cfg-1"
    )
    assert n == 7
    conn.execute.assert_awaited_once()


async def test_list_snapshots_returns_empty_when_no_files() -> None:
    exporter, _, _ = _make_exporter(select_rows=[])
    snapshots = await exporter.list_snapshots(export_config_id="cfg-1")
    assert snapshots == []


async def test_list_snapshots_builds_one_per_file() -> None:
    # First execute() = group-by query returning (file_id, min_chunk_index).
    # Subsequent calls = representative-row fetches per file.
    group_result = MagicMock()
    group_result.all = MagicMock(return_value=[("file-1", 0), ("file-2", 0)])

    rep_result_1 = MagicMock()
    rep_result_1.first = MagicMock(
        return_value=MagicMock(
            metadata_fields={"k": "v"},
            content_fingerprint_hash="fp1",
            source_modified_at=None,
            exported_at=None,
        )
    )
    rep_result_2 = MagicMock()
    rep_result_2.first = MagicMock(
        return_value=MagicMock(
            metadata_fields=None,
            content_fingerprint_hash=None,
            source_modified_at=None,
            exported_at=None,
        )
    )

    exporter, conn, _ = _make_exporter()
    conn.execute = AsyncMock(side_effect=[group_result, rep_result_1, rep_result_2])

    snapshots = await exporter.list_snapshots(export_config_id="cfg-1")

    assert {s.parsed_directory_file_id for s in snapshots} == {"file-1", "file-2"}
    by_id = {s.parsed_directory_file_id: s for s in snapshots}
    assert by_id["file-1"].metadata_fields_hash is not None
    assert by_id["file-1"].content_fingerprint_hash == "fp1"
    assert by_id["file-2"].metadata_fields_hash is None
