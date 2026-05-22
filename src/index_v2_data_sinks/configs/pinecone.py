"""Settings for Pinecone in Index V2.

Platform-managed Pinecone credentials for the default directory-index
destination. Each export config gets its own Pinecone *namespace* within a
shared index, so a single Pinecone project can host an arbitrary number of
tenants without per-tenant infrastructure. Indexes themselves are
dimension-locked, so the exporter provisions one index per embedding model
(see ``services/pinecone.py``).

Mirrors the layout of the other Index V2 sink configs (Turbopuffer,
PGVector, Mongo, Qdrant): a single base set of fields plus per-target-id
override maps for ``api_key`` and ``index_prefix``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PineconeMetric = Literal["cosine", "dotproduct", "euclidean"]
PineconeCloud = Literal["aws", "gcp", "azure"]


class PineconeIndexV2Settings(BaseSettings):
    """Settings for the platform-managed Pinecone project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_by_name=True,
        env_prefix="DEFAULT_INDEX_PINECONE_",
    )

    api_key: SecretStr = Field(description="Platform Pinecone API key.")
    index_prefix: str = Field(
        default="llamacloud",
        description=(
            "Prefix used to derive the platform-managed index name. The "
            "default embedding model keeps ``{prefix}``; non-default models "
            "get a sanitized ``{prefix}-{model}`` suffix because Pinecone "
            "indexes are dimension-locked."
        ),
    )
    cloud: PineconeCloud = Field(
        default="aws",
        description="Cloud provider for serverless index provisioning.",
    )
    region: str = Field(
        default="us-east-1",
        description="Cloud region for serverless index provisioning.",
    )
    metric: PineconeMetric = Field(
        default="cosine",
        description=(
            "Distance metric used when creating new indexes. Pinecone fixes "
            "the metric at index creation time, so changing this only "
            "affects newly-provisioned indexes."
        ),
    )
    timeout: float = Field(
        default=30.0,
        description="Per-request timeout in seconds, forwarded to PineconeAsyncio.",
    )

    api_key_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description=(
            "Per-target-id API key overrides keyed by target id. Values are "
            "raw API keys (handled as secrets)."
        ),
        json_schema_extra={"is_secret": True},
    )
    index_prefix_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description=(
            "Per-target-id index-prefix overrides. Values bypass the global "
            "``index_prefix``."
        ),
    )

    def resolve_api_key(self, target_id: str) -> str:
        if target_id in self.api_key_map:
            return self.api_key_map[target_id]
        return self.api_key.get_secret_value()

    def resolve_index_prefix(self, target_id: str) -> str:
        return self.index_prefix_map.get(target_id, self.index_prefix)


@lru_cache
def get_pinecone_indexv2_settings() -> PineconeIndexV2Settings:
    """Cached platform Pinecone settings."""
    return PineconeIndexV2Settings()  # type: ignore
