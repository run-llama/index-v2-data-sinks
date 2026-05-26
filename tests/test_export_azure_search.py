"""Tests for index_v2_data_sinks.export.azure_search.AzureSearchExporter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from index_v2_data_sinks.export.azure_search import (
    AzureSearchExporter,
    _document_key,
)

from ._helpers import AsyncIter, make_exported


def _make_exporter(*, search_docs: list[dict] | None = None):
    client = MagicMock()
    # ``client.search`` is awaited and returns an async iterable.
    client.search = AsyncMock(return_value=AsyncIter(search_docs or []))
    client.merge_or_upload_documents = AsyncMock()
    client.delete_documents = AsyncMock()

    target_service = MagicMock()
    target_service.ensure_index = AsyncMock(return_value=(client, "index-name"))

    exporter = AzureSearchExporter(target_service=target_service, target_id="DEFAULT")
    return exporter, client


async def test_export_requires_embeddings() -> None:
    exporter, _ = _make_exporter()
    with pytest.raises(ValueError, match="embeddings"):
        await exporter.export([make_exported()], project_id="proj-1", embeddings=None)


async def test_export_empty_outputs_is_noop() -> None:
    exporter, client = _make_exporter()
    await exporter.export([], project_id="proj-1", embeddings=[])
    client.search.assert_not_awaited()
    client.merge_or_upload_documents.assert_not_awaited()


async def test_export_deletes_stale_keys_then_merge_uploads() -> None:
    stale_key = "file-1__9"
    exporter, client = _make_exporter(search_docs=[{"id": stale_key}])
    outputs = [make_exported(chunk_index=0), make_exported(chunk_index=1)]

    await exporter.export(outputs, project_id="proj-1", embeddings=[[0.1], [0.2]])

    # Stale keys deleted first.
    client.delete_documents.assert_awaited_once_with(documents=[{"id": stale_key}])
    # Then the new documents uploaded.
    client.merge_or_upload_documents.assert_awaited_once()
    docs = client.merge_or_upload_documents.call_args.kwargs["documents"]
    assert [d["id"] for d in docs] == [_document_key(o) for o in outputs]
    assert [d["embedding"] for d in docs] == [[0.1], [0.2]]
    assert docs[0]["tenant_id"] == "proj-1"
    assert docs[0]["export_config_id"] == "cfg-1"


async def test_export_skips_delete_when_no_stale_keys() -> None:
    exporter, client = _make_exporter(search_docs=[])
    await exporter.export([make_exported()], project_id="proj-1", embeddings=[[0.1]])
    client.delete_documents.assert_not_awaited()
    client.merge_or_upload_documents.assert_awaited_once()


async def test_export_rejects_row_missing_routing_fields() -> None:
    exporter, _ = _make_exporter()
    bad = make_exported(tenant_id="")
    with pytest.raises(ValueError, match="tenant_id"):
        await exporter.export([bad], project_id="proj-1", embeddings=[[0.1]])


async def test_delete_file_returns_count_and_calls_delete() -> None:
    exporter, client = _make_exporter(
        search_docs=[{"id": "file-1__0"}, {"id": "file-1__1"}]
    )
    n = await exporter.delete_file(
        parsed_directory_file_id="file-1", export_config_id="cfg-1"
    )
    assert n == 2
    client.delete_documents.assert_awaited_once()


async def test_delete_file_returns_zero_when_no_keys() -> None:
    exporter, client = _make_exporter(search_docs=[])
    n = await exporter.delete_file(
        parsed_directory_file_id="missing", export_config_id="cfg-1"
    )
    assert n == 0
    client.delete_documents.assert_not_awaited()


async def test_delete_files_empty_list_is_noop() -> None:
    exporter, client = _make_exporter()
    n = await exporter.delete_files(
        parsed_directory_file_ids=[], export_config_id="cfg-1"
    )
    assert n == 0
    client.search.assert_not_awaited()
    client.delete_documents.assert_not_awaited()


async def test_delete_files_returns_count() -> None:
    exporter, client = _make_exporter(
        search_docs=[{"id": "a__0"}, {"id": "b__0"}, {"id": "b__1"}]
    )
    n = await exporter.delete_files(
        parsed_directory_file_ids=["a", "b"], export_config_id="cfg-1"
    )
    assert n == 3
    client.delete_documents.assert_awaited_once()


async def test_list_snapshots_dedupes_by_file_id() -> None:
    docs = [
        {
            "parsed_directory_file_id": "file-1",
            "metadata_fields_hash": "h1",
            "content_fingerprint_hash": "fp1",
            "source_modified_at": None,
            "exported_at": None,
        },
        # Duplicate file-1 chunk -> should be skipped.
        {
            "parsed_directory_file_id": "file-1",
            "metadata_fields_hash": "h-other",
        },
        {
            "parsed_directory_file_id": "file-2",
            "metadata_fields_hash": None,
            "content_fingerprint_hash": None,
            "source_modified_at": None,
            "exported_at": None,
        },
    ]
    exporter, _ = _make_exporter(search_docs=docs)
    snapshots = await exporter.list_snapshots(export_config_id="cfg-1")
    by_id = {s.parsed_directory_file_id: s for s in snapshots}
    assert set(by_id) == {"file-1", "file-2"}
    # First chunk wins on dedupe.
    assert by_id["file-1"].metadata_fields_hash == "h1"
    assert by_id["file-1"].content_fingerprint_hash == "fp1"
    assert by_id["file-2"].metadata_fields_hash is None
