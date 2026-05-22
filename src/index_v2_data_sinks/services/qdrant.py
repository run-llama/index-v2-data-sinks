"""Qdrant export target: connection resolution and collection lifecycle.

Wraps :class:`QdrantIndexV2Settings` so callers don't need to know about
environment variable layout or per-target overrides. Each target id resolves
to a Qdrant collection; collections are created lazily on first export and
configured with the vector size of the embedding model in use plus payload
indexes on the fields the exporter filters by.
"""

from dataclasses import dataclass
from functools import lru_cache

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from ..configs.qdrant import (
    QdrantIndexV2Settings,
    get_qdrant_indexv2_settings,
)
from ..embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SupportedEmbeddingModel,
    get_embeddings_service,
)

DEFAULT_TARGET_ID = "DEFAULT"

# Payload fields the exporter filters by. Declared centrally so the service
# (creates the index) and the exporter (issues the filtered scroll/delete)
# stay in lockstep.
EXPORT_CONFIG_ID_FIELD = "export_config_id"
TENANT_ID_FIELD = "tenant_id"
PARSED_DIRECTORY_FILE_ID_FIELD = "parsed_directory_file_id"
METADATA_FIELDS_HASH_FIELD = "metadata_fields_hash"

QDRANT_INDEX_V2_NOT_CONFIGURED_MESSAGE = (
    "Qdrant index V2 is not configured for this environment. "
    "Set DEFAULT_INDEX_QDRANT_URL (and optionally DEFAULT_INDEX_QDRANT_API_KEY, "
    "DEFAULT_INDEX_QDRANT_COLLECTION_PREFIX)."
)


class QdrantIndexV2NotConfiguredError(RuntimeError):
    """Raised when Qdrant index V2 settings are missing required values.

    HTTP boundary callers should translate this into a 503 so the response
    is actionable instead of a generic 500 with a pydantic trace.
    """


@dataclass(frozen=True)
class QdrantConnectionTarget:
    """Resolved Qdrant connection coordinates for a single export target."""

    url: str
    api_key: str | None
    collection: str


def _sanitize_model_suffix(model: SupportedEmbeddingModel) -> str:
    """Turn a model identifier into a safe Qdrant collection suffix."""
    return (
        model.replace(":", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("@", "")
        .lower()
    )


def _resolve_collection_name(
    base: str, embedding_model: SupportedEmbeddingModel | None
) -> str:
    """Per-embedding-model collection name.

    The default embedding model keeps the base name for backward
    compatibility; other models get a deterministic suffix so two configs
    that emit vectors of different dimensions don't collide.
    """
    if embedding_model is None or embedding_model == DEFAULT_EMBEDDING_MODEL:
        return base
    return f"{base}_{_sanitize_model_suffix(embedding_model)}"


# Bounded cache because BYO-credentials wiring can push multiple
# (url, api_key, prefer_grpc) tuples through here; 128 leaves plenty of
# headroom for the platform-only deployment while still capping growth.
_CLIENT_CACHE_MAXSIZE = 128


@lru_cache(maxsize=_CLIENT_CACHE_MAXSIZE)
def _get_qdrant_async_client(
    url: str, api_key: str | None, prefer_grpc: bool, timeout: int | None
) -> AsyncQdrantClient:
    """Return a cached AsyncQdrantClient keyed by connection params."""
    return AsyncQdrantClient(
        url=url, api_key=api_key, prefer_grpc=prefer_grpc, timeout=timeout
    )


class QdrantService:
    """Resolves a target id to concrete Qdrant connection parameters.

    Accepts an optional :class:`QdrantIndexV2Settings` for testability;
    when omitted the cached singleton from
    :func:`get_qdrant_indexv2_settings` is used.
    """

    def __init__(self, settings: QdrantIndexV2Settings | None = None) -> None:
        self._settings = settings
        # (url, collection) pairs we've already ensured a collection on.
        self._collections_ensured: set[tuple[str, str]] = set()

    @property
    def settings(self) -> QdrantIndexV2Settings:
        if self._settings is None:
            try:
                self._settings = get_qdrant_indexv2_settings()
            except ValidationError as e:
                raise QdrantIndexV2NotConfiguredError(
                    QDRANT_INDEX_V2_NOT_CONFIGURED_MESSAGE
                ) from e
        return self._settings

    def resolve(
        self,
        target_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> QdrantConnectionTarget:
        """Return connection coordinates for ``target_id``."""
        base = self.settings.resolve_collection(target_id)
        return QdrantConnectionTarget(
            url=self.settings.resolve_url(target_id),
            api_key=self.settings.resolve_api_key(target_id),
            collection=_resolve_collection_name(base, embedding_model),
        )

    def get_client(self, target_id: str) -> AsyncQdrantClient:
        """Return a cached AsyncQdrantClient for ``target_id``."""
        target = self.resolve(target_id)
        return _get_qdrant_async_client(
            target.url,
            target.api_key,
            self.settings.prefer_grpc,
            self.settings.timeout,
        )

    async def ensure_collection(
        self,
        target_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> tuple[AsyncQdrantClient, str]:
        """Ensure the target's collection exists and return ``(client, name)``.

        Creates the collection with a vector of the model's native size and
        ``Cosine`` distance, and provisions keyword payload indexes on the
        scoping fields the exporter filters by. All operations are
        idempotent.
        """
        target = self.resolve(target_id, embedding_model)
        client = self.get_client(target_id)
        cache_key = (target.url, target.collection)
        if cache_key in self._collections_ensured:
            return client, target.collection

        dims = get_embeddings_service().resolve_dims(embedding_model)
        if not await client.collection_exists(target.collection):
            await client.create_collection(
                collection_name=target.collection,
                vectors_config=qmodels.VectorParams(
                    size=dims, distance=qmodels.Distance.COSINE
                ),
            )

        # Payload indexes are idempotent in Qdrant — re-creating an existing
        # one is a no-op, so we don't need a separate "was this already
        # created" cache per field.
        for field_name in (
            EXPORT_CONFIG_ID_FIELD,
            TENANT_ID_FIELD,
            PARSED_DIRECTORY_FILE_ID_FIELD,
        ):
            await client.create_payload_index(
                collection_name=target.collection,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

        self._collections_ensured.add(cache_key)
        return client, target.collection


qdrant_service = QdrantService()


def get_qdrant_service() -> QdrantService:
    """Return the module-level singleton."""
    return qdrant_service
