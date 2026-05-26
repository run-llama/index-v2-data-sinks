from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _normalize_pg_uri(uri: str) -> str:
    """Rewrite common PostgreSQL URI forms to the async SQLAlchemy form.

    SQLAlchemy's dialect is ``postgresql`` (not ``postgres``) and we want
    the async ``asyncpg`` driver. Accept the common shorthand forms users
    paste from cloud providers and normalize them.
    """
    for src in ("postgres://", "postgresql://"):
        if uri.startswith(src):
            return "postgresql+asyncpg://" + uri[len(src) :]
    return uri


class PGVectorIndexV2Settings(BaseSettings):
    """Settings for PostgreSQL + pgvector in Index V2."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_by_name=True,
        env_prefix="DEFAULT_INDEX_PGVECTOR_",
    )

    # Name of the PostgreSQL schema to write into (e.g. "public").
    schema_name: str = Field(
        description="Name of the PostgreSQL schema",
        default="public",
    )

    # Async SQLAlchemy URI, e.g. "postgresql+asyncpg://user:pass@host:5432/db".
    uri: str = Field(
        description="Connection URI for PostgreSQL (asyncpg driver)",
        json_schema_extra={"is_secret": True},
    )

    # Table name within the PostgreSQL database used to store chunks.
    table: str = Field(description="Table name to use for storing exported outputs")

    uri_map: dict[str, str] = Field(
        description="Map for PostgreSQL URIs by target ID",
        default_factory=lambda: {},
        json_schema_extra={"is_secret": True},
    )
    schema_map: dict[str, str] = Field(
        description="Map for PostgreSQL schemas by target ID",
        default_factory=lambda: {},
    )
    tables_map: dict[str, str] = Field(
        description="Map for PostgreSQL table names by target ID",
        default_factory=lambda: {},
    )

    def get_redacted_uri(self) -> str:
        # Preserve the driver prefix (e.g. "postgresql+asyncpg://") and redact
        # the rest, which can contain credentials.
        if "://" in self.uri:
            prefix, rest = self.uri.split("://", 1)
            return f"{prefix}://" + "*" * len(rest)
        return "*" * len(self.uri)

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, v: str) -> str:
        return _normalize_pg_uri(v)

    @field_validator("uri_map")
    @classmethod
    def _validate_uri_map(cls, v: dict[str, str]) -> dict[str, str]:
        return {k: _normalize_pg_uri(val) for k, val in v.items()}

    def resolve_uri(self, k: str) -> str:
        """Resolve the URI for the given target ID."""
        return self.uri_map.get(k, self.uri)

    def resolve_schema_name(self, k: str) -> str:
        """Resolve the schema name for the given target ID."""
        return self.schema_map.get(k, self.schema_name)

    def resolve_table_name(self, k: str) -> str:
        """Resolve the table name for the given target ID."""
        return self.tables_map.get(k, self.table)


@lru_cache
def get_pgvector_indexv2_settings() -> PGVectorIndexV2Settings:
    """Get cached PGVector IndexV2 settings."""
    # fields are loaded from env variables, ignore call issues reports
    return PGVectorIndexV2Settings()  # type: ignore
