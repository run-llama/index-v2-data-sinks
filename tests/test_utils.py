"""Tests for index_v2_data_sinks.utils."""

from __future__ import annotations

from pydantic import BaseModel

from index_v2_data_sinks.utils import compute_metadata_hash


class _Meta(BaseModel):
    a: int
    b: str


def test_compute_metadata_hash_none_returns_none() -> None:
    assert compute_metadata_hash(None) is None


def test_compute_metadata_hash_empty_dict_returns_none() -> None:
    assert compute_metadata_hash({}) is None


def test_compute_metadata_hash_dict_is_deterministic_and_key_order_independent() -> (
    None
):
    h1 = compute_metadata_hash({"a": 1, "b": "x"})
    h2 = compute_metadata_hash({"b": "x", "a": 1})
    assert h1 is not None
    assert h1 == h2
    # SHA-256 hex digest length
    assert len(h1) == 64


def test_compute_metadata_hash_dict_changes_when_values_change() -> None:
    assert compute_metadata_hash({"a": 1}) != compute_metadata_hash({"a": 2})


def test_compute_metadata_hash_basemodel_matches_dump() -> None:
    m = _Meta(a=1, b="x")
    assert compute_metadata_hash(m) == compute_metadata_hash({"a": 1, "b": "x"})
