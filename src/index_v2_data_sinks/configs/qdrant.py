"""Settings for Qdrant in Index V2.

Platform-managed Qdrant credentials for the default directory-index
destination. Each export config gets its own Qdrant collection, derived from
the export config ID and an optional prefix, so a single Qdrant cluster can
host an arbitrary number of tenants without per-tenant infrastructure.

Mirrors the layout of the other Index V2 sink configs (Turbopuffer,
PGVector, Mongo): a single base set of fields plus per-target-id override
maps for ``url``, ``api_key``, and ``collection``.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantIndexV2Settings(BaseSettings):
    """Settings for the platform-managed Qdrant cluster."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_by_name=True,
        env_prefix="DEFAULT_INDEX_QDRANT_",
    )

    url: str = Field(
        description=(
            "Qdrant HTTP(S) URL (e.g. http://localhost:6333 or "
            "https://xyz-example.eu-central.aws.cloud.qdrant.io)."
        )
    )
    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Optional Qdrant API key. Required for Qdrant Cloud; can be "
            "omitted when running against an unauthenticated local instance."
        ),
    )
    collection_prefix: str = Field(
        default="llamacloud",
        description=(
            "Prefix prepended to every platform-managed collection name. "
            "Final collection = ``{prefix}-{target_id}``."
        ),
    )
    prefer_grpc: bool = Field(
        default=False,
        description=(
            "Use the gRPC transport for write-heavy paths if true. The "
            "default REST transport is sufficient for ingest at this scale."
        ),
    )
    timeout: int | None = Field(
        default=None,
        description="Optional per-request timeout in seconds.",
    )

    url_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description="Per-target-id Qdrant URL overrides keyed by target id.",
    )
    api_key_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description=(
            "Per-target-id Qdrant API key overrides. Values are the raw API "
            "keys (handled as secrets)."
        ),
        json_schema_extra={"is_secret": True},
    )
    collection_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description=(
            "Per-target-id collection overrides. Keys are target ids; values "
            "are the full collection name to use, bypassing "
            "``collection_prefix``."
        ),
    )

    def resolve_url(self, target_id: str) -> str:
        return self.url_map.get(target_id, self.url)

    def resolve_api_key(self, target_id: str) -> str | None:
        if target_id in self.api_key_map:
            return self.api_key_map[target_id]
        return self.api_key.get_secret_value() if self.api_key is not None else None

    def resolve_collection(self, target_id: str) -> str:
        if target_id in self.collection_map:
            return self.collection_map[target_id]
        return f"{self.collection_prefix}-{target_id}"


@lru_cache
def get_qdrant_indexv2_settings() -> QdrantIndexV2Settings:
    """Cached platform Qdrant settings."""
    return QdrantIndexV2Settings()  # type: ignore
