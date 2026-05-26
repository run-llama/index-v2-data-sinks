"""Azure AI Search export target: connection resolution and index lifecycle.

Wraps :class:`AzureSearchIndexV2Settings` so callers don't need to know
about environment variable layout or per-target overrides. Each
``target_id`` resolves to an Azure AI Search *service* (endpoint +
credential); within that service, the service provisions one index per
embedding model (Azure AI Search indexes are schema-locked, including
vector dimension) and the exporter writes per-export-config records into
a shared index, scoped by an ``export_config_id`` filter.

Auth: prefers an admin API key from settings when provided, otherwise
falls back to ``DefaultAzureCredential`` (AAD). The credential is
constructed lazily and cached so the AAD code path doesn't pay token
acquisition cost when API-key auth is in use.
"""

from dataclasses import dataclass
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes import models as azmodels
from azure.search.documents.indexes.aio import SearchIndexClient
from pydantic import ValidationError

from ..configs.azure_search import (
    AzureSearchIndexV2Settings,
    get_azure_search_indexv2_settings,
)
from ..embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SupportedEmbeddingModel,
    get_embeddings_service,
)

DEFAULT_TARGET_ID = "DEFAULT"

# Document field names the exporter filters on / surfaces in snapshots.
# Centralized so the service (declares the field schema) and the
# exporter (issues the OData filters) stay in lockstep on names.
#
# Azure AI Search field names must start with a letter and contain only
# letters, digits, and underscores. The names below comply; if you add
# new fields, validate them against that rule.
KEY_FIELD = "id"
EXPORT_CONFIG_ID_FIELD = "export_config_id"
TENANT_ID_FIELD = "tenant_id"
PARSED_DIRECTORY_FILE_ID_FIELD = "parsed_directory_file_id"
METADATA_FIELDS_HASH_FIELD = "metadata_fields_hash"
EMBEDDING_FIELD = "embedding"

# Names for the per-index vector-search configuration. Azure AI Search
# requires the index definition to reference a named algorithm and a
# named profile; they're index-local so they can't collide across
# indexes.
_HNSW_CONFIG_NAME = "default-hnsw"
_VECTOR_PROFILE_NAME = "default-vector-profile"

AZURE_SEARCH_INDEX_V2_NOT_CONFIGURED_MESSAGE = (
    "Azure AI Search index V2 is not configured for this environment. "
    "Set DEFAULT_INDEX_AZURE_SEARCH_ENDPOINT (and either "
    "DEFAULT_INDEX_AZURE_SEARCH_API_KEY for key auth, or make sure "
    "DefaultAzureCredential can resolve a token for AAD auth)."
)


class AzureSearchIndexV2NotConfiguredError(RuntimeError):
    """Raised when Azure AI Search index V2 settings are missing required values.

    HTTP boundary callers should translate this into a 503 so the
    response is actionable instead of a generic 500 with a pydantic
    trace.
    """


@dataclass(frozen=True)
class AzureSearchConnectionTarget:
    """Resolved Azure AI Search connection coordinates for a single target."""

    endpoint: str
    api_key: str | None
    index_name: str


def _sanitize_model_suffix(model: SupportedEmbeddingModel) -> str:
    """Turn a model identifier into a safe Azure index-name suffix.

    Azure AI Search index names must be lowercase, start and end with an
    alphanumeric, contain only letters/digits/hyphens, and not have
    consecutive hyphens. We normalize via the same rules used by the
    Pinecone service (which has the same character class), then collapse
    repeated hyphens.
    """
    cleaned = (
        model.replace(":", "-")
        .replace("/", "-")
        .replace(".", "-")
        .replace("_", "-")
        .replace("@", "")
        .lower()
    )
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _resolve_index_name(
    base: str, embedding_model: SupportedEmbeddingModel | None
) -> str:
    """Per-embedding-model index name.

    The default embedding model keeps the base name for backward
    compatibility; other models get a deterministic suffix so two
    configs that emit vectors of different dimensions don't collide on
    the same (dimension-locked) Azure index.
    """
    if embedding_model is None or embedding_model == DEFAULT_EMBEDDING_MODEL:
        return base
    return f"{base}-{_sanitize_model_suffix(embedding_model)}"


# Cached AAD credential. We instantiate it lazily because constructing
# ``DefaultAzureCredential`` is non-trivial (env probe + IMDS probe on
# Azure VMs) and most deployments will use API-key auth. ``lru_cache``
# without a key argument is fine here because the credential is shared
# process-wide; it does not embed any per-target state.
@lru_cache(maxsize=1)
def _get_default_credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()


# Bounded client caches keyed by (endpoint, api_key, index_name) where
# applicable. The Azure SDK's ``SearchClient`` and ``SearchIndexClient``
# both hold onto an httpx pool, so reusing them across calls avoids
# per-request connection churn. 128 entries leaves plenty of headroom
# for BYO-credentials wiring while still capping growth.
_CLIENT_CACHE_MAXSIZE = 128


def _credential(api_key: str | None) -> AzureKeyCredential | AsyncTokenCredential:
    """Return the credential object the SDK clients expect.

    API key wins when present; otherwise fall back to ``DefaultAzureCredential``.
    """
    if api_key is not None:
        return AzureKeyCredential(api_key)
    return _get_default_credential()


@lru_cache(maxsize=_CLIENT_CACHE_MAXSIZE)
def _get_search_index_client(endpoint: str, api_key: str | None) -> SearchIndexClient:
    """Cached control-plane SearchIndexClient.

    Used for index lifecycle (create/check). One per (endpoint, key);
    distinct keys against the same endpoint get distinct clients so
    rotating an api_key key map entry doesn't surface stale auth.
    """
    return SearchIndexClient(endpoint=endpoint, credential=_credential(api_key))


@lru_cache(maxsize=_CLIENT_CACHE_MAXSIZE)
def _get_search_client(
    endpoint: str, api_key: str | None, index_name: str
) -> SearchClient:
    """Cached data-plane SearchClient.

    Used for document upload/delete/search. Keyed by index because the
    SDK ties a SearchClient to a single index name.
    """
    return SearchClient(
        endpoint=endpoint, index_name=index_name, credential=_credential(api_key)
    )


def _build_index_definition(name: str, dims: int, metric: str) -> azmodels.SearchIndex:
    """Build the Azure AI Search index schema for chunk storage.

    Field choices mirror the payload the exporter writes (see
    ``export/azure_search.py``):

    - ``id``: the unique key. Synthesized from
      ``parsed_directory_file_id`` and ``chunk_index`` so re-exports of
      the same chunk overwrite (idempotent upload).
    - filterable scoping fields (``tenant_id``, ``export_config_id``,
      ``parsed_directory_file_id``) — needed for per-config snapshots
      and per-file deletes.
    - filterable / retrievable provenance + change-detection fields
      (``content_fingerprint_hash``, ``source_modified_at``,
      ``exported_at``, ``metadata_fields_hash``).
    - searchable + retrievable ``content`` so the index doubles as a
      hybrid (BM25 + vector) retrieval target without a second hop.
    - vector ``embedding`` with the model's native dimension, bound to a
      single HNSW profile. Cosine is the typical metric and is the
      default; alternatives are surfaced via config.
    """
    # Map our metric label to the Azure SDK's enum. Names match Azure's
    # convention; ``dotProduct`` is camelCase upstream.
    metric_enum = {
        "cosine": azmodels.VectorSearchAlgorithmMetric.COSINE,
        "dotProduct": azmodels.VectorSearchAlgorithmMetric.DOT_PRODUCT,
        "euclidean": azmodels.VectorSearchAlgorithmMetric.EUCLIDEAN,
    }[metric]

    fields: list[azmodels.SearchField] = [
        # Key. Must be a non-nullable string and the only field with
        # ``key=True``.
        azmodels.SimpleField(
            name=KEY_FIELD,
            type=azmodels.SearchFieldDataType.String.value,
            key=True,
            filterable=True,
        ),
        azmodels.SimpleField(
            name=TENANT_ID_FIELD,
            type=azmodels.SearchFieldDataType.String.value,
            filterable=True,
        ),
        azmodels.SimpleField(
            name=EXPORT_CONFIG_ID_FIELD,
            type=azmodels.SearchFieldDataType.String.value,
            filterable=True,
        ),
        azmodels.SimpleField(
            name=PARSED_DIRECTORY_FILE_ID_FIELD,
            type=azmodels.SearchFieldDataType.String.value,
            filterable=True,
        ),
        azmodels.SimpleField(
            name="source_file_id",
            type=azmodels.SearchFieldDataType.String.value,
            filterable=True,
        ),
        azmodels.SimpleField(
            name="chunk_index",
            type=azmodels.SearchFieldDataType.Int32.value,
            filterable=True,
        ),
        azmodels.SimpleField(
            name="chunk_start_char",
            type=azmodels.SearchFieldDataType.Int32.value,
        ),
        azmodels.SimpleField(
            name="chunk_end_char",
            type=azmodels.SearchFieldDataType.Int32.value,
        ),
        azmodels.SimpleField(
            name="chunk_token_count",
            type=azmodels.SearchFieldDataType.Int32.value,
        ),
        azmodels.SimpleField(
            name="page_range_start",
            type=azmodels.SearchFieldDataType.Int32.value,
            filterable=True,
        ),
        azmodels.SimpleField(
            name="page_range_end",
            type=azmodels.SearchFieldDataType.Int32.value,
            filterable=True,
        ),
        azmodels.SimpleField(
            name=METADATA_FIELDS_HASH_FIELD,
            type=azmodels.SearchFieldDataType.String.value,
            filterable=True,
        ),
        azmodels.SimpleField(
            name="content_fingerprint_hash",
            type=azmodels.SearchFieldDataType.String.value,
            filterable=True,
        ),
        # Azure stores datetimes as ``DateTimeOffset``. We emit ISO 8601
        # strings on write; the SDK parses them.
        azmodels.SimpleField(
            name="source_modified_at",
            type=azmodels.SearchFieldDataType.DateTimeOffset.value,
            filterable=True,
            sortable=True,
        ),
        azmodels.SimpleField(
            name="exported_at",
            type=azmodels.SearchFieldDataType.DateTimeOffset.value,
            filterable=True,
            sortable=True,
        ),
        # Free-text content. Searchable so callers can hybrid-search via
        # BM25 + vector in a single request. Marked retrievable so RAG
        # callers don't need a second hop to fetch chunk text.
        azmodels.SearchableField(
            name="content",
            type=azmodels.SearchFieldDataType.String,
        ),
        # Attachments are JSON-encoded into a single string. Azure
        # supports complex types but they're awkward to evolve; the
        # round-trip via ``json.loads`` is cheap on the read side.
        azmodels.SimpleField(
            name="attachments_json",
            type=azmodels.SearchFieldDataType.String.value,
        ),
        # Custom user metadata is also JSON-encoded into a single
        # string. Keeping it as one field avoids index-schema churn
        # whenever the caller adds a new metadata key, at the cost of
        # not being able to OData-filter on individual entries. The
        # ``metadata_fields_hash`` field above covers the change-
        # detection use case that snapshots need.
        azmodels.SimpleField(
            name="metadata_fields_json",
            type=azmodels.SearchFieldDataType.String.value,
        ),
        # Vector field. ``Collection(Edm.Single)`` is the standard type
        # for dense embeddings; ``vector_search_dimensions`` locks the
        # dimension at index-creation time, which is exactly why we
        # provision one index per embedding model.
        azmodels.SearchField(
            name=EMBEDDING_FIELD,
            type="Collection(Edm.Single)",
            searchable=True,
            vector_search_dimensions=dims,
            vector_search_profile_name=_VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = azmodels.VectorSearch(
        algorithms=[
            azmodels.HnswAlgorithmConfiguration(
                name=_HNSW_CONFIG_NAME,
                parameters=azmodels.HnswParameters(metric=metric_enum),
            ),
        ],
        profiles=[
            azmodels.VectorSearchProfile(
                name=_VECTOR_PROFILE_NAME,
                algorithm_configuration_name=_HNSW_CONFIG_NAME,
            ),
        ],
    )

    return azmodels.SearchIndex(name=name, fields=fields, vector_search=vector_search)


class AzureSearchService:
    """Resolves a target id to concrete Azure AI Search connection parameters.

    Accepts an optional :class:`AzureSearchIndexV2Settings` for
    testability; when omitted the cached singleton from
    :func:`get_azure_search_indexv2_settings` is used.
    """

    def __init__(self, settings: AzureSearchIndexV2Settings | None = None) -> None:
        self._settings = settings
        # (endpoint, index_name) pairs we've already ensured exist remotely.
        self._indexes_ensured: set[tuple[str, str]] = set()

    @property
    def settings(self) -> AzureSearchIndexV2Settings:
        if self._settings is None:
            try:
                self._settings = get_azure_search_indexv2_settings()
            except ValidationError as e:
                raise AzureSearchIndexV2NotConfiguredError(
                    AZURE_SEARCH_INDEX_V2_NOT_CONFIGURED_MESSAGE
                ) from e
        return self._settings

    def resolve(
        self,
        target_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> AzureSearchConnectionTarget:
        """Return connection coordinates for ``target_id``."""
        base = self.settings.resolve_index_prefix(target_id)
        return AzureSearchConnectionTarget(
            endpoint=self.settings.resolve_endpoint(target_id),
            api_key=self.settings.resolve_api_key(target_id),
            index_name=_resolve_index_name(base, embedding_model),
        )

    def get_index_client(self, target_id: str) -> SearchIndexClient:
        """Return a cached control-plane SearchIndexClient for ``target_id``."""
        target = self.resolve(target_id)
        return _get_search_index_client(target.endpoint, target.api_key)

    async def ensure_index(
        self,
        target_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> tuple[SearchClient, str]:
        """Ensure the target's index exists and return ``(client, name)``.

        Creates an Azure AI Search index with the model's native vector
        dimension and the configured metric on first call, then caches
        the resulting data-plane :class:`SearchClient`. Subsequent calls
        hit only the in-memory cache.

        Index existence is checked via ``get_index`` (404 → create);
        Azure does not have an ``if-not-exists`` create endpoint, and
        racing two creates returns 409 which we treat as success.
        """
        target = self.resolve(target_id, embedding_model)
        cache_key = (target.endpoint, target.index_name)

        if cache_key not in self._indexes_ensured:
            index_client = self.get_index_client(target_id)
            # ``get_index`` raises ``ResourceNotFoundError`` on 404; any
            # other error (auth, transient) should propagate so callers
            # can decide how to handle it.
            from azure.core.exceptions import (
                ResourceExistsError,
                ResourceNotFoundError,
            )

            exists = True
            try:
                await index_client.get_index(target.index_name)
            except ResourceNotFoundError:
                exists = False
            if not exists:
                dims = get_embeddings_service().resolve_dims(embedding_model)
                definition = _build_index_definition(
                    target.index_name, dims, self.settings.metric
                )
                try:
                    await index_client.create_index(definition)
                except ResourceExistsError:
                    # Another concurrent caller won the create race. The
                    # resulting index has the same dims + metric (both
                    # are deterministic functions of inputs we control),
                    # so this is safe to treat as success.
                    pass
            self._indexes_ensured.add(cache_key)

        return (
            _get_search_client(target.endpoint, target.api_key, target.index_name),
            target.index_name,
        )


azure_search_service = AzureSearchService()


def get_azure_search_service() -> AzureSearchService:
    """Return the module-level singleton."""
    return azure_search_service
