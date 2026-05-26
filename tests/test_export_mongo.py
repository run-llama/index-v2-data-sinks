"""Tests for index_v2_data_sinks.export.mongo.MongoExporter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from index_v2_data_sinks.export.mongo import MongoExporter

from ._helpers import AsyncIter, make_exported


def _make_exporter() -> tuple[MongoExporter, MagicMock]:
    coll = MagicMock()
    coll.delete_many = AsyncMock(return_value=MagicMock(deleted_count=3))
    coll.insert_many = AsyncMock()

    target_service = MagicMock()
    target_service.get_collection = AsyncMock(return_value=coll)

    exporter = MongoExporter(target_service=target_service, target_id="DEFAULT")
    return exporter, coll


async def test_export_requires_embeddings() -> None:
    exporter, _ = _make_exporter()
    with pytest.raises(ValueError, match="embeddings"):
        await exporter.export([make_exported()], project_id="proj-1", embeddings=None)


async def test_export_empty_outputs_is_noop() -> None:
    exporter, coll = _make_exporter()
    await exporter.export([], project_id="proj-1", embeddings=[])
    coll.delete_many.assert_not_awaited()
    coll.insert_many.assert_not_awaited()


async def test_export_deletes_then_inserts_for_file() -> None:
    exporter, coll = _make_exporter()
    outputs = [
        make_exported(chunk_index=0),
        make_exported(chunk_index=1),
    ]
    embeddings = [[0.1], [0.2]]

    await exporter.export(outputs, project_id="proj-1", embeddings=embeddings)

    coll.delete_many.assert_awaited_once_with(
        {"parsed_directory_file_id": "file-1", "export_config_id": "cfg-1"}
    )
    coll.insert_many.assert_awaited_once()
    inserted = coll.insert_many.call_args.args[0]
    assert len(inserted) == 2


async def test_delete_file_returns_deleted_count() -> None:
    exporter, coll = _make_exporter()
    n = await exporter.delete_file(
        parsed_directory_file_id="file-1", export_config_id="cfg-1"
    )
    assert n == 3
    coll.delete_many.assert_awaited_once_with(
        {"parsed_directory_file_id": "file-1", "export_config_id": "cfg-1"}
    )


async def test_delete_files_empty_list_is_noop() -> None:
    exporter, coll = _make_exporter()
    n = await exporter.delete_files(
        parsed_directory_file_ids=[], export_config_id="cfg-1"
    )
    assert n == 0
    coll.delete_many.assert_not_awaited()


async def test_delete_files_uses_in_query() -> None:
    exporter, coll = _make_exporter()
    n = await exporter.delete_files(
        parsed_directory_file_ids=["a", "b"], export_config_id="cfg-1"
    )
    assert n == 3
    coll.delete_many.assert_awaited_once_with(
        {
            "parsed_directory_file_id": {"$in": ["a", "b"]},
            "export_config_id": "cfg-1",
        }
    )


async def test_list_snapshots_builds_snapshots_from_aggregate() -> None:
    exporter, coll = _make_exporter()
    docs = [
        {
            "_id": "file-1",
            "metadata_fields": {"k": "v"},
            "content_fingerprint_hash": "fp1",
            "source_modified_at": None,
            "exported_at": None,
        },
        {
            "_id": "file-2",
            "metadata_fields": None,
            "content_fingerprint_hash": None,
            "source_modified_at": None,
            "exported_at": None,
        },
    ]
    coll.aggregate = AsyncMock(return_value=AsyncIter(docs))

    snapshots = await exporter.list_snapshots(export_config_id="cfg-1")

    assert {s.parsed_directory_file_id for s in snapshots} == {"file-1", "file-2"}
    by_id = {s.parsed_directory_file_id: s for s in snapshots}
    # file-1 has metadata -> hash is set; file-2 has none -> hash is None.
    assert by_id["file-1"].metadata_fields_hash is not None
    assert by_id["file-2"].metadata_fields_hash is None
    assert by_id["file-1"].content_fingerprint_hash == "fp1"
    coll.aggregate.assert_awaited_once()
