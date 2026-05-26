"""PGVector export target: connection resolution, schema management, and table lifecycle.

Wraps the PGVectorIndexV2Settings config so callers don't need to know
about environment variable layout or the DEFAULT / map-based resolution
logic. Each target is identified by a string key (e.g. ``"DEFAULT"``).

Also owns the PostgreSQL table lifecycle: ensuring the ``vector`` extension
is enabled, creating per-embedding-model tables with a properly-sized
``VECTOR`` column, and creating supporting indexes (HNSW for ANN search
and a composite btree for snapshot/diff queries).
"""

from dataclasses import dataclass
from functools import lru_cache

from pgvector.sqlalchemy import Vector
from pydantic import ValidationError
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    text,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ..configs.pgvector import (
    PGVectorIndexV2Settings,
    get_pgvector_indexv2_settings,
)
from ..embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SupportedEmbeddingModel,
    get_embeddings_service,
)

EXPORT_CONFIG_FILE_INDEX_SUFFIX = "_export_config_file_idx"
EMBEDDING_INDEX_SUFFIX = "_embedding_hnsw_idx"

DEFAULT_TARGET_ID = "DEFAULT"

PGVECTOR_INDEX_V2_NOT_CONFIGURED_MESSAGE = (
    "PGVector index V2 is not configured for this environment. "
    "Set DEFAULT_INDEX_PGVECTOR_URI and DEFAULT_INDEX_PGVECTOR_TABLE."
)


class PGVectorIndexV2NotConfiguredError(RuntimeError):
    """Raised when PGVector index V2 settings are missing required values.

    Callers at the HTTP boundary should translate this into a 503 so the
    response is actionable instead of a generic 500 with a pydantic trace.
    """


@dataclass(frozen=True)
class PGVectorConnectionTarget:
    """Resolved PGVector connection coordinates for a single export destination."""

    uri: str
    schema_name: str
    table: str


@lru_cache
def _get_engine(uri: str) -> AsyncEngine:
    """Return a cached ``AsyncEngine`` for the given URI."""
    return create_async_engine(uri, pool_pre_ping=True)


def _sanitize_model_suffix(model: SupportedEmbeddingModel) -> str:
    """Turn a model identifier into a safe SQL identifier suffix."""
    return (
        model.replace(":", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("@", "")
        .lower()
    )


def _resolve_table_name(
    base_table: str, embedding_model: SupportedEmbeddingModel | None
) -> str:
    """Compute the per-embedding-model table name.

    Mirrors the Mongo search-index naming: the default embedding model
    keeps the base name for backward compatibility, while non-default
    models get a deterministic suffix.
    """
    if embedding_model is None or embedding_model == DEFAULT_EMBEDDING_MODEL:
        return base_table
    return f"{base_table}_{_sanitize_model_suffix(embedding_model)}"


class PGVectorService:
    """Resolves a target ID to concrete PGVector connection parameters.

    Accepts an optional ``PGVectorIndexV2Settings`` for testability; when
    omitted the cached singleton from ``get_pgvector_indexv2_settings()``
    is used.
    """

    def __init__(self, settings: PGVectorIndexV2Settings | None = None) -> None:
        self._settings = settings
        self._extension_ensured: set[str] = set()
        self._tables_ensured: dict[str, Table] = {}

    @property
    def settings(self) -> PGVectorIndexV2Settings:
        if self._settings is None:
            try:
                self._settings = get_pgvector_indexv2_settings()
            except ValidationError as e:
                raise PGVectorIndexV2NotConfiguredError(
                    PGVECTOR_INDEX_V2_NOT_CONFIGURED_MESSAGE
                ) from e
        return self._settings

    def resolve(
        self,
        target_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> PGVectorConnectionTarget:
        """Return the connection coordinates for *target_id*."""
        base_table = self.settings.resolve_table_name(target_id)
        return PGVectorConnectionTarget(
            uri=self.settings.resolve_uri(target_id),
            schema_name=self.settings.resolve_schema_name(target_id),
            table=_resolve_table_name(base_table, embedding_model),
        )

    def get_engine(self, target_id: str) -> AsyncEngine:
        """Return a cached ``AsyncEngine`` for *target_id*."""
        return _get_engine(self.resolve(target_id).uri)

    async def get_table(
        self,
        target_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> tuple[AsyncEngine, Table]:
        """Return a ready-to-use ``(engine, Table)`` with schema/indexes ensured."""
        target = self.resolve(target_id, embedding_model)
        engine = self.get_engine(target_id)

        cache_key = f"{target.uri}|{target.schema_name}|{target.table}"
        if cache_key in self._tables_ensured:
            return engine, self._tables_ensured[cache_key]

        dims = get_embeddings_service().resolve_dims(embedding_model)
        table = self._build_table(target.schema_name, target.table, dims)

        await self._ensure_extension(engine, target.uri)
        await self._ensure_schema(engine, target.schema_name)
        await self._ensure_table(engine, table)
        await self._ensure_indexes(engine, table)

        self._tables_ensured[cache_key] = table
        return engine, table

    # ------------------------------------------------------------------
    # Schema / index management (internal)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_table(schema_name: str, table_name: str, dims: int) -> Table:
        metadata = MetaData(schema=schema_name)
        return Table(
            table_name,
            metadata,
            Column("id", String, primary_key=True),
            Column("content", String, nullable=False),
            Column("chunk_index", Integer, nullable=False),
            Column("chunk_start_char", Integer, nullable=True),
            Column("chunk_end_char", Integer, nullable=True),
            Column("chunk_token_count", Integer, nullable=True),
            Column("parsed_directory_file_id", String, nullable=False),
            Column("source_file_id", String, nullable=True),
            Column("page_range_start", Integer, nullable=False),
            Column("page_range_end", Integer, nullable=False),
            Column("tenant_id", String, nullable=False),
            Column("export_config_id", String, nullable=False),
            Column("metadata_fields", JSON, nullable=True),
            Column("content_fingerprint_hash", String, nullable=True),
            Column("source_modified_at", DateTime(timezone=True), nullable=True),
            Column("exported_at", DateTime(timezone=True), nullable=True),
            Column("attachments", JSON, nullable=False),
            Column("embedding", Vector(dims), nullable=False),
            Index(
                f"{table_name}{EXPORT_CONFIG_FILE_INDEX_SUFFIX}",
                "export_config_id",
                "parsed_directory_file_id",
            ),
        )

    async def _ensure_extension(self, engine: AsyncEngine, uri: str) -> None:
        if uri in self._extension_ensured:
            return
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self._extension_ensured.add(uri)

    @staticmethod
    async def _ensure_schema(engine: AsyncEngine, schema_name: str) -> None:
        async with engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))

    @staticmethod
    async def _ensure_table(engine: AsyncEngine, table: Table) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(table.metadata.create_all, tables=[table])

    @staticmethod
    async def _ensure_indexes(engine: AsyncEngine, table: Table) -> None:
        schema = table.schema or "public"
        index_name = f"{table.name}{EMBEDDING_INDEX_SUFFIX}"
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                    f'ON "{schema}"."{table.name}" '
                    f"USING hnsw (embedding vector_cosine_ops)"
                )
            )


pgvector_service = PGVectorService()


def get_pgvector_service() -> PGVectorService:
    """Return the module-level singleton."""
    return pgvector_service
