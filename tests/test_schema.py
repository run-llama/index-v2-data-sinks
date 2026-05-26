"""Tests for index_v2_data_sinks.schema."""

from __future__ import annotations

import pytest

from index_v2_data_sinks.schema import (
    AttachmentRef,
    Chunk,
    ExportedOutputV1,
    ParsedPage,
    ParseSyncOutputV1,
    _build_page_boundaries,
    _overlapping_pages,
    is_metadata_value,
)


def _make_output(pages: list[ParsedPage]) -> ParseSyncOutputV1:
    return ParseSyncOutputV1(version="v1", pages=pages)


def test_is_metadata_value_accepts_scalars_and_str_lists() -> None:
    assert is_metadata_value(None)
    assert is_metadata_value("x")
    assert is_metadata_value(1)
    assert is_metadata_value(1.5)
    assert is_metadata_value(True)
    assert is_metadata_value(["a", "b"])


def test_is_metadata_value_rejects_mixed_lists_and_objects() -> None:
    assert not is_metadata_value(["a", 1])
    assert not is_metadata_value({"a": 1})
    assert not is_metadata_value(object())


def test_build_page_boundaries_skips_empty_pages_and_tracks_offsets() -> None:
    p1 = ParsedPage(page_number=1, text="hello")
    p_empty = ParsedPage(page_number=2, text=None)
    p3 = ParsedPage(page_number=3, text="world")
    boundaries, segments = _build_page_boundaries(_make_output([p1, p_empty, p3]))

    assert segments == ["hello", "world"]
    assert len(boundaries) == 2
    assert boundaries[0].page_number == 1
    assert boundaries[1].page_number == 3
    # Boundaries should be non-overlapping and the second should start after
    # the first, accounting for the "\n" separator.
    assert boundaries[0].end_char <= boundaries[1].start_char


def test_build_page_boundaries_raises_when_no_content() -> None:
    with pytest.raises(ValueError):
        _build_page_boundaries(_make_output([ParsedPage(page_number=1, text=None)]))


def test_overlapping_pages_returns_only_intersecting_pages() -> None:
    p1 = ParsedPage(page_number=1, text="hello")  # 5 chars -> [0, 5)
    p2 = ParsedPage(page_number=2, text="world")  # +1 sep -> [6, 11)
    boundaries, _ = _build_page_boundaries(_make_output([p1, p2]))

    # Chunk entirely inside page 1.
    chunk_p1 = Chunk(
        index=0,
        text="hel",  # codespell:ignore
        start_char_idx=0,
        end_char_idx=3,
    )
    assert [p.page_number for p in _overlapping_pages(chunk_p1, boundaries)] == [1]

    # Chunk spanning the boundary.
    chunk_both = Chunk(index=0, text="lo\nwo", start_char_idx=3, end_char_idx=8)
    assert [p.page_number for p in _overlapping_pages(chunk_both, boundaries)] == [
        1,
        2,
    ]


def test_overlapping_pages_falls_back_to_all_when_indices_missing() -> None:
    p1 = ParsedPage(page_number=1, text="hello")
    p2 = ParsedPage(page_number=2, text="world")
    boundaries, _ = _build_page_boundaries(_make_output([p1, p2]))
    chunk = Chunk(index=0, text="anything")  # no start/end
    assert _overlapping_pages(chunk, boundaries) == boundaries


def test_exported_output_build_populates_page_range_and_attachments() -> None:
    att = AttachmentRef(
        type="screenshot", attachment_name="screenshots/p1.jpg", source_id="src-1"
    )
    p1 = ParsedPage(page_number=1, text="hello", attachments={"screenshot": att})
    p2 = ParsedPage(page_number=2, text="world")
    out = _make_output([p1, p2])

    chunk = Chunk(
        index=0,
        text="hel",  # codespell:ignore
        start_char_idx=0,
        end_char_idx=3,
    )
    exported = ExportedOutputV1.build(
        output=out,
        project_id="proj-1",
        config_id="cfg-1",
        chunk=chunk,
        parsed_directory_file_id="file-1",
        metadata={"k": "v"},
        source_file_id="src-1",
    )

    assert exported.content == "hel"  # codespell:ignore
    assert exported.chunk_index == 0
    assert exported.page_range_start == 1
    assert exported.page_range_end == 1
    assert exported.tenant_id == "proj-1"
    assert exported.export_config_id == "cfg-1"
    assert exported.metadata_fields == {"k": "v"}
    assert exported.source_file_id == "src-1"
    assert len(exported.attachments) == 1
    assert exported.attachments[0].attachment_name == "screenshots/p1.jpg"
    assert exported.exported_at is not None


def test_exported_output_build_rejects_empty_pages() -> None:
    out = ParseSyncOutputV1(version="v1", pages=[])
    with pytest.raises(ValueError):
        ExportedOutputV1.build(
            output=out,
            project_id="p",
            config_id="c",
            chunk=Chunk(index=0, text="x"),
            parsed_directory_file_id="f",
        )


def test_exported_output_to_raw_bson_doc_with_embedding_round_trips() -> None:
    from bson import decode

    p1 = ParsedPage(page_number=1, text="hello")
    out = _make_output([p1])
    chunk = Chunk(index=0, text="hello", start_char_idx=0, end_char_idx=5)
    exported = ExportedOutputV1.build(
        output=out,
        project_id="proj-1",
        config_id="cfg-1",
        chunk=chunk,
        parsed_directory_file_id="file-1",
    )
    raw = exported.to_raw_bson_doc_with_embedding([0.1, 0.2, 0.3])
    decoded = decode(bytes(raw.raw))
    assert decoded["content"] == "hello"
    assert decoded["embedding"] == [0.1, 0.2, 0.3]
    assert decoded["tenant_id"] == "proj-1"
