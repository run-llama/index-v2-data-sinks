"""Tests for index_v2_data_sinks.chunking.

Uses the real chonkie chunkers (already a project dependency) but with
short inputs so tests stay fast.
"""

from __future__ import annotations

from index_v2_data_sinks.chunking import ChunkerService
from index_v2_data_sinks.schema import Chunk


def test_chunker_service_returns_list_of_chunks_for_string() -> None:
    service = ChunkerService()
    text = "Hello world. This is a sentence. And another one here."
    chunks = service.chunk(text, strategy="sentence")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    # Indices are 0-based and contiguous.
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Every chunk has text.
    assert all(c.text for c in chunks)


def test_chunker_service_returns_list_of_lists_for_batch() -> None:
    service = ChunkerService()
    inputs = ["Hello world.", "Another doc here."]
    batches = service.chunk(inputs, strategy="sentence")
    assert isinstance(batches, list)
    assert len(batches) == 2
    for batch in batches:
        assert isinstance(batch, list)
        assert all(isinstance(c, Chunk) for c in batch)


def test_chunker_service_fast_strategy_runs() -> None:
    service = ChunkerService()
    chunks = service.chunk("Hello world. Some more text here.", strategy="fast")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
