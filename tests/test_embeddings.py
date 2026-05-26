"""Tests for index_v2_data_sinks.embeddings."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from index_v2_data_sinks import embeddings as embeddings_mod
from index_v2_data_sinks.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_TO_DIMENSIONS,
    EmbeddingsService,
    check_api_key,
)


def test_check_api_key_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        check_api_key("not-a-provider")


def test_check_api_key_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key found"):
        check_api_key("openai")


def test_check_api_key_present_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    check_api_key("openai")  # should not raise


def test_check_api_key_gemini_accepts_either_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    check_api_key("gemini")  # should not raise


def test_check_api_key_cloudflare_requires_all_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError, match="Missing required"):
        check_api_key("cloudflare")

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    check_api_key("cloudflare")  # both present -> ok


def test_resolve_dims_returns_known_dimension() -> None:
    service = EmbeddingsService()
    assert (
        service.resolve_dims("openai:text-embedding-3-small")
        == EMBEDDING_MODEL_TO_DIMENSIONS["openai:text-embedding-3-small"]
    )
    # Defaults to DEFAULT_EMBEDDING_MODEL when None passed.
    assert (
        service.resolve_dims(None)
        == EMBEDDING_MODEL_TO_DIMENSIONS[DEFAULT_EMBEDDING_MODEL]
    )


async def test_embed_single_string_returns_first_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    service = EmbeddingsService()
    fake_response = SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
    service._client = SimpleNamespace(aembed=AsyncMock(return_value=fake_response))

    result = await service.embed("hello")
    assert result == [0.1, 0.2, 0.3]
    service._client.aembed.assert_awaited_once()


async def test_embed_list_returns_all_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    service = EmbeddingsService()
    fake_response = SimpleNamespace(embeddings=[[0.1], [0.2]])
    service._client = SimpleNamespace(aembed=AsyncMock(return_value=fake_response))

    result = await service.embed(["a", "b"], model="openai:text-embedding-3-small")
    assert result == [[0.1], [0.2]]


async def test_embed_unknown_provider_raises_before_calling_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = EmbeddingsService()
    aembed = AsyncMock()
    service._client = SimpleNamespace(aembed=aembed)

    with pytest.raises(ValueError):
        await service.embed("hello", model="openai:text-embedding-3-small")
    aembed.assert_not_awaited()


def test_module_level_embeddings_service_is_cached() -> None:
    a = embeddings_mod.get_embeddings_service()
    b = embeddings_mod.get_embeddings_service()
    assert a is b
