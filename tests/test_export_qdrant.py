"""Tests for index_v2_data_sinks.export.qdrant.QdrantExporter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.http import models as qmodels

from index_v2_data_sinks.export.qdrant import QdrantExporter, _point_id

from ._helpers import make_exported


def _make_exporter(*, count: int = 2):
    client = MagicMock()
    client.delete = AsyncMock()
    client.upsert = AsyncMock()
    client.count = AsyncMock(return_value=MagicMock(count=count))
    client.scroll = AsyncMock(return_value=([], None))

    target_service = MagicMock()
    target_service.ensure_collection = AsyncMock(
        return_value=(client, "llamacloud-DEFAULT")
    )

    exporter = QdrantExporter(target_service=target_service, target_id="DEFAULT")
    return exporter, client


async def test_export_requires_embeddings() -> None:
    exporter, _ = _make_exporter()
    with pytest.raises(ValueError, match="embeddings"):
        await exporter.export([make_exported()], project_id="proj-1", embeddings=None)


async def test_export_empty_outputs_is_noop() -> None:
    exporter, client = _make_exporter()
    await exporter.export([], project_id="proj-1", embeddings=[])
    client.delete.assert_not_awaited()
    client.upsert.assert_not_awaited()


async def test_export_deletes_then_upserts_points() -> None:
    exporter, client = _make_exporter()
    outputs = [make_exported(chunk_index=0), make_exported(chunk_index=1)]
    await exporter.export(outputs, project_id="proj-1", embeddings=[[0.1], [0.2]])

    client.delete.assert_awaited_once()
    delete_kwargs = client.delete.call_args.kwargs
    assert delete_kwargs["collection_name"] == "llamacloud-DEFAULT"
    assert isinstance(delete_kwargs["points_selector"], qmodels.FilterSelector)

    client.upsert.assert_awaited_once()
    upsert_kwargs = client.upsert.call_args.kwargs
    points = upsert_kwargs["points"]
    assert [p.id for p in points] == [_point_id(o) for o in outputs]
    assert [p.vector for p in points] == [[0.1], [0.2]]
    # Each payload carries the routing fields used by filters.
    for p in points:
        assert p.payload["tenant_id"] == "proj-1"
        assert p.payload["export_config_id"] == "cfg-1"


async def test_export_rejects_row_missing_tenant_id() -> None:
    exporter, _ = _make_exporter()
    bad = make_exported(tenant_id="")
    with pytest.raises(ValueError, match="tenant_id"):
        await exporter.export([bad], project_id="proj-1", embeddings=[[0.1]])


async def test_delete_file_returns_count_then_deletes() -> None:
    exporter, client = _make_exporter(count=5)
    n = await exporter.delete_file(
        parsed_directory_file_id="file-1", export_config_id="cfg-1"
    )
    assert n == 5
    client.count.assert_awaited_once()
    client.delete.assert_awaited_once()


async def test_delete_files_empty_list_is_noop() -> None:
    exporter, client = _make_exporter()
    n = await exporter.delete_files(
        parsed_directory_file_ids=[], export_config_id="cfg-1"
    )
    assert n == 0
    client.count.assert_not_awaited()
    client.delete.assert_not_awaited()


async def test_delete_files_uses_match_any() -> None:
    exporter, client = _make_exporter(count=4)
    n = await exporter.delete_files(
        parsed_directory_file_ids=["a", "b"], export_config_id="cfg-1"
    )
    assert n == 4
    count_filter = client.count.call_args.kwargs["count_filter"]
    # The filter should embed a MatchAny condition with the supplied ids.
    conds = count_filter.must
    has_match_any = any(
        getattr(c, "match", None) is not None
        and isinstance(getattr(c.match, "any", None), list)
        and c.match.any == ["a", "b"]
        for c in conds
    )
    assert has_match_any


async def test_list_snapshots_dedupes_by_file_id() -> None:
    exporter, client = _make_exporter()

    # Two pages: first has two records (one duplicate file-id), second empty.
    rec1 = MagicMock()
    rec1.payload = {
        "parsed_directory_file_id": "file-1",
        "metadata_fields_hash": "h1",
        "content_fingerprint_hash": "fp1",
        "source_modified_at": None,
        "exported_at": None,
    }
    rec_dup = MagicMock()
    rec_dup.payload = {
        "parsed_directory_file_id": "file-1",
        "metadata_fields_hash": "h-other",
    }
    rec2 = MagicMock()
    rec2.payload = {
        "parsed_directory_file_id": "file-2",
        "metadata_fields_hash": None,
        "content_fingerprint_hash": None,
    }
    client.scroll = AsyncMock(
        side_effect=[
            ([rec1, rec_dup, rec2], "offset-1"),
            ([], None),
        ]
    )

    snapshots = await exporter.list_snapshots(export_config_id="cfg-1")
    by_id = {s.parsed_directory_file_id: s for s in snapshots}
    assert set(by_id) == {"file-1", "file-2"}
    # First record wins on dedupe.
    assert by_id["file-1"].metadata_fields_hash == "h1"
    assert by_id["file-1"].content_fingerprint_hash == "fp1"
    assert by_id["file-2"].metadata_fields_hash is None
