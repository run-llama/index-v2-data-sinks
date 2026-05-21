"""Settings for Turbopuffer in Index V2.

Platform-managed Turbopuffer credentials for the default directory-index
destination. Each export config gets its own tpuf namespace, derived from
the export config ID and an optional prefix, so a single tpuf account can
host an arbitrary number of tenants without per-tenant infrastructure.

Custom user-supplied credentials are handled separately via the data sink
layer (``CloudTurbopufferVectorStore``); this settings class only covers
the platform's default account.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class TurbopufferIndexV2Settings(BaseSettings):
    """Settings for the platform-managed Turbopuffer account."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_by_name=True,
        env_prefix="DEFAULT_INDEX_TPUF_",
    )

    api_key: SecretStr = Field(description="Platform Turbopuffer API key")
    region: str = Field(
        default="aws-us-east-1",
        description="Default tpuf region for namespaces created by the platform",
    )
    namespace_prefix: str = Field(
        default="llamacloud",
        description=(
            "Prefix prepended to every platform-managed namespace name. "
            "Final namespace = ``{prefix}-{target_id}``."
        ),
    )
    region_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description="Per-target-id region overrides keyed by target id (defaults to ``region``).",
    )
    namespace_map: dict[str, str] = Field(
        default_factory=lambda: {},
        description=(
            "Per-target-id namespace overrides. Keys are target ids; values are the full "
            "namespace name to use, bypassing ``namespace_prefix``."
        ),
    )

    def resolve_region(self, target_id: str) -> str:
        return self.region_map.get(target_id, self.region)

    def resolve_namespace(self, target_id: str) -> str:
        if target_id in self.namespace_map:
            return self.namespace_map[target_id]
        return f"{self.namespace_prefix}-{target_id}"


@lru_cache
def get_turbopuffer_indexv2_settings() -> TurbopufferIndexV2Settings:
    """Cached platform Turbopuffer settings."""
    return TurbopufferIndexV2Settings()  # type: ignore
