"""Tests for index_v2_data_sinks.configs.* settings classes."""

from __future__ import annotations

import pytest

from index_v2_data_sinks.configs.mongo import (
    MongoDBIndexV2Settings,
    get_mongodb_indexv2_settings,
)
from index_v2_data_sinks.configs.pinecone import PineconeIndexV2Settings
from index_v2_data_sinks.configs.qdrant import (
    QdrantIndexV2Settings,
    get_qdrant_indexv2_settings,
)


def _build_mongo_settings(**overrides) -> MongoDBIndexV2Settings:
    base = dict(db="default_db", uri="mongodb://u:p@host:27017", collection="default")
    base.update(overrides)
    return MongoDBIndexV2Settings(**base)  # type: ignore[arg-type]


def test_mongo_resolve_falls_back_to_defaults() -> None:
    s = _build_mongo_settings()
    assert s.resolve_uri("anything") == "mongodb://u:p@host:27017"
    assert s.resolve_database_name("anything") == "default_db"
    assert s.resolve_collection_name("anything") == "default"


def test_mongo_resolve_uses_maps_when_present() -> None:
    s = _build_mongo_settings(
        uri_map={"tenant-1": "mongodb://other:other@h:27017"},
        db_map={"tenant-1": "tenant_db"},
        collections_map={"tenant-1": "tenant_coll"},
    )
    assert s.resolve_uri("tenant-1") == "mongodb://other:other@h:27017"
    assert s.resolve_database_name("tenant-1") == "tenant_db"
    assert s.resolve_collection_name("tenant-1") == "tenant_coll"
    # Unknown keys still fall back.
    assert s.resolve_uri("other") == s.uri


def test_mongo_get_redacted_uri_masks_credentials() -> None:
    s = _build_mongo_settings(uri="mongodb://user:pass@host:27017/db")
    redacted = s.get_redacted_uri()
    assert redacted.startswith("mongodb://")
    assert "user" not in redacted
    assert "pass" not in redacted
    # mongodb+srv prefix is preserved.
    s2 = _build_mongo_settings(uri="mongodb+srv://user:pass@host/db")
    assert s2.get_redacted_uri().startswith("mongodb+srv://")


def test_mongo_settings_cached() -> None:
    get_mongodb_indexv2_settings.cache_clear()
    a = get_mongodb_indexv2_settings()
    b = get_mongodb_indexv2_settings()
    assert a is b


def test_qdrant_resolve_url_collection_and_api_key() -> None:
    s = QdrantIndexV2Settings(  # type: ignore[arg-type]
        url="http://default:6333",
        api_key="default-key",  # type: ignore[arg-type]
        collection_prefix="llamacloud",
        url_map={"t1": "http://t1:6333"},
        api_key_map={"t1": "t1-key"},
        collection_map={"t2": "explicit_collection"},
    )
    # url / api_key overrides
    assert s.resolve_url("t1") == "http://t1:6333"
    assert s.resolve_url("missing") == "http://default:6333"
    assert s.resolve_api_key("t1") == "t1-key"
    assert s.resolve_api_key("missing") == "default-key"

    # collection: explicit override wins, otherwise prefix-{target_id}
    assert s.resolve_collection("t2") == "explicit_collection"
    assert s.resolve_collection("t3") == "llamacloud-t3"


def test_qdrant_resolve_api_key_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Avoid picking up .env / pre-set DEFAULT_INDEX_QDRANT_API_KEY from the
    # repo's local environment.
    monkeypatch.delenv("DEFAULT_INDEX_QDRANT_API_KEY", raising=False)
    s = QdrantIndexV2Settings(
        url="http://x:6333",
        api_key=None,
        _env_file=None,  # type: ignore[call-arg]
    )
    assert s.resolve_api_key("anything") is None


def test_qdrant_settings_cached() -> None:
    get_qdrant_indexv2_settings.cache_clear()
    a = get_qdrant_indexv2_settings()
    b = get_qdrant_indexv2_settings()
    assert a is b


def test_pinecone_resolve_api_key_and_index_prefix() -> None:
    s = PineconeIndexV2Settings(  # type: ignore[arg-type]
        api_key="default-key",  # type: ignore[arg-type]
        index_prefix="llamacloud",
        api_key_map={"t1": "t1-key"},
        index_prefix_map={"t2": "custom"},
    )
    assert s.resolve_api_key("t1") == "t1-key"
    assert s.resolve_api_key("missing") == "default-key"
    assert s.resolve_index_prefix("t2") == "custom"
    assert s.resolve_index_prefix("missing") == "llamacloud"


def test_pinecone_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEFAULT_INDEX_PINECONE_API_KEY", raising=False)
    with pytest.raises(Exception):
        PineconeIndexV2Settings(_env_file=None)  # type: ignore[call-arg]
