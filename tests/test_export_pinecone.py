"""Tests for index_v2_data_sinks.export.pinecone.PineconeExporter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from index_v2_data_sinks.export.pinecone import (
    PineconeExporter,
    _file_id_prefix,
    _point_id,
)

from ._helpers import make_exported


def _list_response(ids: list[str], next_token: str | None = None) -> MagicMock:
    """Build a fake ``list_paginated`` response with the given ids."""
    resp = MagicMock()
    resp.vectors = [MagicMock(id=i) for i in ids]
    resp.pagination = MagicMock(next=next_token) if next_token else None
    return resp


def _make_exporter():
    index = MagicMock()
    index.list_paginated = AsyncMock(return_value=_list_response([]))
    index.delete = AsyncMock()
    index.upsert = AsyncMock()
    index.fetch = AsyncMock()

    target_service = MagicMock()
    target_service.ensure_index = AsyncMock(return_value=(index, "my-index"))

    exporter = PineconeExporter(target_service=target_service, target_id="DEFAULT")
    return exporter, index


async def test_export_requires_embeddings() -> None:
    exporter, _ = _make_exporter()
    with pytest.raises(ValueError, match="embeddings"):
        await exporter.export([make_exported()], project_id="proj-1", embeddings=None)


async def test_export_empty_outputs_is_noop() -> None:
    exporter, index = _make_exporter()
    await exporter.export([], project_id="proj-1", embeddings=[])
    index.upsert.assert_not_awaited()
    index.delete.assert_not_awaited()


async def test_export_lists_stale_ids_deletes_then_upserts() -> None:
    exporter, index = _make_exporter()
    stale_id = "file-1#9"
    index.list_paginated = AsyncMock(return_value=_list_response([stale_id]))

    outputs = [make_exported(chunk_index=0), make_exported(chunk_index=1)]
    await exporter.export(outputs, project_id="proj-1", embeddings=[[0.1], [0.2]])

    # List-by-prefix used the file's prefix.
    list_kwargs = index.list_paginated.call_args.kwargs
    assert list_kwargs["prefix"] == _file_id_prefix("file-1")
    assert list_kwargs["namespace"] == "cfg-1"

    index.delete.assert_awaited_once_with(ids=[stale_id], namespace="cfg-1")
    index.upsert.assert_awaited_once()
    upsert_kwargs = index.upsert.call_args.kwargs
    vectors = upsert_kwargs["vectors"]
    assert [v["id"] for v in vectors] == [_point_id(o) for o in outputs]
    assert [v["values"] for v in vectors] == [[0.1], [0.2]]
    # Metadata carries the routing fields plus content.
    assert vectors[0]["metadata"]["tenant_id"] == "proj-1"
    assert vectors[0]["metadata"]["export_config_id"] == "cfg-1"
    assert vectors[0]["metadata"]["content"] == "chunk-0"


async def test_export_skips_delete_when_no_stale_ids() -> None:
    exporter, index = _make_exporter()
    # Empty list response -> no delete call, still upserts.
    outputs = [make_exported()]
    await exporter.export(outputs, project_id="proj-1", embeddings=[[0.1]])
    index.delete.assert_not_awaited()
    index.upsert.assert_awaited_once()


async def test_delete_file_returns_count() -> None:
    exporter, index = _make_exporter()
    index.list_paginated = AsyncMock(
        return_value=_list_response(["file-1#0", "file-1#1"])
    )
    n = await exporter.delete_file(
        parsed_directory_file_id="file-1", export_config_id="cfg-1"
    )
    assert n == 2
    index.delete.assert_awaited_once_with(
        ids=["file-1#0", "file-1#1"], namespace="cfg-1"
    )


async def test_delete_file_returns_zero_when_no_ids() -> None:
    exporter, index = _make_exporter()  # default: empty list response
    n = await exporter.delete_file(
        parsed_directory_file_id="missing", export_config_id="cfg-1"
    )
    assert n == 0
    index.delete.assert_not_awaited()


async def test_delete_files_empty_list_is_noop() -> None:
    exporter, index = _make_exporter()
    n = await exporter.delete_files(
        parsed_directory_file_ids=[], export_config_id="cfg-1"
    )
    assert n == 0
    index.list_paginated.assert_not_awaited()
    index.delete.assert_not_awaited()


async def test_delete_files_lists_per_file_and_aggregates_count() -> None:
    exporter, index = _make_exporter()
    # Return different ids per file by cycling responses.
    index.list_paginated = AsyncMock(
        side_effect=[
            _list_response(["a#0", "a#1"]),
            _list_response(["b#0"]),
        ]
    )
    n = await exporter.delete_files(
        parsed_directory_file_ids=["a", "b"], export_config_id="cfg-1"
    )
    assert n == 3
    assert index.delete.await_count == 2


async def test_list_snapshots_uses_fetch_for_representatives() -> None:
    exporter, index = _make_exporter()
    # list_paginated returns ids for two files; chunk #0 is preferred.
    index.list_paginated = AsyncMock(
        return_value=_list_response(["file-1#5", "file-1#0", "file-2#3"])
    )

    fetch_resp = MagicMock()
    fetch_resp.vectors = {
        "file-1#0": MagicMock(
            metadata={
                "parsed_directory_file_id": "file-1",
                "metadata_fields_hash": "h1",
                "content_fingerprint_hash": "fp1",
            }
        ),
        "file-2#3": MagicMock(
            metadata={
                "parsed_directory_file_id": "file-2",
                "metadata_fields_hash": None,
                "content_fingerprint_hash": None,
            }
        ),
    }
    index.fetch = AsyncMock(return_value=fetch_resp)

    snapshots = await exporter.list_snapshots(export_config_id="cfg-1")
    by_id = {s.parsed_directory_file_id: s for s in snapshots}
    assert set(by_id) == {"file-1", "file-2"}
    assert by_id["file-1"].metadata_fields_hash == "h1"
    assert by_id["file-1"].content_fingerprint_hash == "fp1"

    # Representative selection: chunk #0 must have been preferred over #5.
    fetched_ids = index.fetch.call_args.kwargs["ids"]
    assert "file-1#0" in fetched_ids
    assert "file-1#5" not in fetched_ids
