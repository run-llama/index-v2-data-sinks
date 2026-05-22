"""Pinecone export target: connection resolution and index lifecycle.

Wraps :class:`PineconeIndexV2Settings` so callers don't need to know about
environment variable layout or per-target overrides. Each ``target_id``
resolves to a Pinecone *project* (api key); within that project, the
service provisions one index per embedding model (Pinecone indexes are
dimension-locked, so different embedding models cannot share an index)
and the exporter writes per-export-config records into separate Pinecone
*namespaces* within those indexes.
"""

from dataclasses import dataclass
from functools import lru_cache

from pinecone import PineconeAsyncio, ServerlessSpec
from pinecone.db_data import _IndexAsyncio as AsyncIndex
from pydantic import ValidationError

from ..configs.pinecone import (
    PineconeIndexV2Settings,
    get_pinecone_indexv2_settings,
)
from ..embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SupportedEmbeddingModel,
    get_embeddings_service,
)

DEFAULT_TARGET_ID = "DEFAULT"

# Payload metadata fields the exporter filters on / surfaces in snapshots.
# Centralized so the exporter and any future indexing helpers stay in
# lockstep on key names.
EXPORT_CONFIG_ID_FIELD = "export_config_id"
TENANT_ID_FIELD = "tenant_id"
PARSED_DIRECTORY_FILE_ID_FIELD = "parsed_directory_file_id"
METADATA_FIELDS_HASH_FIELD = "metadata_fields_hash"

PINECONE_INDEX_V2_NOT_CONFIGURED_MESSAGE = (
    "Pinecone index V2 is not configured for this environment. "
    "Set DEFAULT_INDEX_PINECONE_API_KEY (and optionally "
    "DEFAULT_INDEX_PINECONE_INDEX_PREFIX, DEFAULT_INDEX_PINECONE_CLOUD, "
    "DEFAULT_INDEX_PINECONE_REGION)."
)


class PineconeIndexV2NotConfiguredError(RuntimeError):
    """Raised when Pinecone index V2 settings are missing required values."""


@dataclass(frozen=True)
class PineconeConnectionTarget:
    """Resolved Pinecone connection coordinates for a single export target."""

    api_key: str
    index_name: str


def _sanitize_model_suffix(model: SupportedEmbeddingModel) -> str:
    """Turn a model identifier into a safe Pinecone index-name suffix.

    Pinecone index names are restricted to lowercase letters, digits, and
    ``-``; everything else gets collapsed to ``-``.
    """
    cleaned = (
        model.replace(":", "-")
        .replace("/", "-")
        .replace(".", "-")
        .replace("_", "-")
        .replace("@", "")
        .lower()
    )
    # Collapse repeated hyphens that result from the substitutions above so
    # the final name stays compact and predictable.
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _resolve_index_name(
    base: str, embedding_model: SupportedEmbeddingModel | None
) -> str:
    """Per-embedding-model index name.

    The default embedding model keeps the base name for backward
    compatibility; other models get a deterministic suffix so two configs
    that emit vectors of different dimensions don't collide on the same
    (dimension-locked) Pinecone index.
    """
    if embedding_model is None or embedding_model == DEFAULT_EMBEDDING_MODEL:
        return base
    return f"{base}-{_sanitize_model_suffix(embedding_model)}"


# Bounded cache keyed by api_key — one PineconeAsyncio control-plane client
# per project. 128 entries leaves plenty of headroom for BYO-credentials
# wiring while still capping growth.
_CLIENT_CACHE_MAXSIZE = 128


@lru_cache(maxsize=_CLIENT_CACHE_MAXSIZE)
def _get_pinecone_async_client(api_key: str, timeout: float) -> PineconeAsyncio:
    """Cached control-plane PineconeAsyncio client.

    Process-lifetime: the cache holds onto the underlying httpx pool; no
    explicit close is needed because the process exiting tears it down.
    """
    return PineconeAsyncio(api_key=api_key, timeout=timeout)


# Separate cache for data-plane index clients. Pinecone's AsyncIndex
# manages its own HTTP client distinct from the control-plane client, and
# is keyed by (api_key, index host). We cache by (api_key, index_name)
# because the control-plane client transparently resolves name -> host.
_INDEX_CACHE: dict[tuple[str, str], AsyncIndex] = {}


class PineconeService:
    """Resolves a target id to concrete Pinecone connection parameters.

    Accepts an optional :class:`PineconeIndexV2Settings` for testability;
    when omitted the cached singleton from
    :func:`get_pinecone_indexv2_settings` is used.
    """

    def __init__(self, settings: PineconeIndexV2Settings | None = None) -> None:
        self._settings = settings
        # (api_key, index_name) pairs we've already ensured exist remotely.
        self._indexes_ensured: set[tuple[str, str]] = set()

    @property
    def settings(self) -> PineconeIndexV2Settings:
        if self._settings is None:
            try:
                self._settings = get_pinecone_indexv2_settings()
            except ValidationError as e:
                raise PineconeIndexV2NotConfiguredError(
                    PINECONE_INDEX_V2_NOT_CONFIGURED_MESSAGE
                ) from e
        return self._settings

    def resolve(
        self,
        target_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> PineconeConnectionTarget:
        """Return connection coordinates for ``target_id``."""
        base = self.settings.resolve_index_prefix(target_id)
        return PineconeConnectionTarget(
            api_key=self.settings.resolve_api_key(target_id),
            index_name=_resolve_index_name(base, embedding_model),
        )

    def get_control_client(self, target_id: str) -> PineconeAsyncio:
        """Return a cached control-plane Pinecone client for ``target_id``."""
        target = self.resolve(target_id)
        return _get_pinecone_async_client(target.api_key, self.settings.timeout)

    async def ensure_index(
        self,
        target_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> tuple[AsyncIndex, str]:
        """Ensure the target's index exists and return ``(index, name)``.

        Creates a serverless Pinecone index with the model's native vector
        dimension and the configured metric/cloud/region on first call,
        then caches the resulting data-plane :class:`AsyncIndex` client.
        Subsequent calls hit only the in-memory cache.
        """
        target = self.resolve(target_id, embedding_model)
        cache_key = (target.api_key, target.index_name)

        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None and cache_key in self._indexes_ensured:
            return cached, target.index_name

        control = self.get_control_client(target_id)

        if cache_key not in self._indexes_ensured:
            if not await control.has_index(target.index_name):
                dims = get_embeddings_service().resolve_dims(embedding_model)
                # Pinecone fixes (dimension, metric) at creation time; for
                # existing indexes we trust whatever they were created with
                # rather than racing to "fix" them.
                await control.create_index(
                    name=target.index_name,
                    dimension=dims,
                    metric=self.settings.metric,
                    spec=ServerlessSpec(
                        cloud=self.settings.cloud, region=self.settings.region
                    ),
                )
            self._indexes_ensured.add(cache_key)

        if cached is None:
            cached = await control.index(name=target.index_name)
            _INDEX_CACHE[cache_key] = cached

        return cached, target.index_name


pinecone_service = PineconeService()


def get_pinecone_service() -> PineconeService:
    """Return the module-level singleton."""
    return pinecone_service
