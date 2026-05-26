"""Settings for Azure AI Search in Index V2.

Platform-managed Azure AI Search credentials for the default
directory-index destination. An Azure AI Search *service* (endpoint)
hosts many *indexes*; indexes are schema-locked, including the vector
dimension, so the exporter provisions one index per embedding model
(see ``services/azure_search.py``).

Azure AI Search has no first-class "namespace" primitive inside an
index, so per-export-config isolation is done by writing
``export_config_id`` onto every document and filtering on it for
snapshots and deletes (OData ``$filter``). This mirrors the way the
Mongo/PGVector/Qdrant sinks scope by field, while the per-model index
shape mirrors Pinecone (which is also dimension-locked).

Mirrors the layout of the other Index V2 sink configs: a single base
set of fields plus per-target-id override maps for ``api_key`` and
``index_prefix``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Azure AI Search exposes two distance metrics that make sense for
# arbitrary embedding models: cosine and dot-product. Euclidean is also
# supported but rarely useful for normalized embeddings. The metric is
# fixed at index-creation time, so changing it only affects newly
# provisioned indexes.
AzureSearchMetric = Literal["cosine", "dotProduct", "euclidean"]


class AzureSearchIndexV2Settings(BaseSettings):
    """Settings for the platform-managed Azure AI Search service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_by_name=True,
        env_prefix="DEFAULT_INDEX_AZURE_SEARCH_",
    )

    endpoint: str = Field(
        description=(
            "Azure AI Search service endpoint, e.g. "
            "``https://my-service.search.windows.net``."
        )
    )
    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Admin API key for the Azure AI Search service. When omitted, "
            "the service falls back to ``DefaultAzureCredential`` (AAD), "
            "which requires the running identity to hold the "
            "``Search Index Data Contributor`` role on the service."
        ),
    )
    index_prefix: str = Field(
        default="llamacloud",
        description=(
            "Prefix used to derive the platform-managed index name. The "
            "default embedding model keeps ``{prefix}``; non-default models "
            "get a sanitized ``{prefix}-{model}`` suffix because Azure AI "
            "Search indexes are dimension-locked."
        ),
    )
    metric: AzureSearchMetric = Field(
        default="cosine",
        description=(
            "Distance metric used when creating new indexes. Azure AI "
            "Search fixes this at index creation time, so changing this "
            "only affects newly-provisioned indexes."
        ),
    )

    api_key_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description=(
            "Per-target-id admin-API-key overrides keyed by target id. "
            "Values are raw API keys (handled as secrets)."
        ),
        json_schema_extra={"is_secret": True},
    )
    endpoint_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description=(
            "Per-target-id endpoint overrides. Values bypass the global "
            "``endpoint`` and let a single deployment fan out across "
            "multiple Azure Search services."
        ),
    )
    index_prefix_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description=(
            "Per-target-id index-prefix overrides. Values bypass the "
            "global ``index_prefix``."
        ),
    )

    def resolve_endpoint(self, target_id: str) -> str:
        return self.endpoint_map.get(target_id, self.endpoint)

    def resolve_api_key(self, target_id: str) -> str | None:
        # The per-target override always wins, even over a globally-set
        # key. Returning ``None`` is meaningful: it signals "fall back to
        # AAD" to the service layer.
        if target_id in self.api_key_map:
            return self.api_key_map[target_id]
        return self.api_key.get_secret_value() if self.api_key is not None else None

    def resolve_index_prefix(self, target_id: str) -> str:
        return self.index_prefix_map.get(target_id, self.index_prefix)


@lru_cache
def get_azure_search_indexv2_settings() -> AzureSearchIndexV2Settings:
    """Cached platform Azure AI Search settings."""
    return AzureSearchIndexV2Settings()  # type: ignore
