"""MongoDB export target: connection resolution, index management, and document insertion.

Wraps the MongoDBIndexV2Settings config so callers don't need to know
about environment variable layout or the DEFAULT / map-based resolution
logic. Each target is identified by a string key (e.g. ``"DEFAULT"``).

Also owns the MongoDB collection lifecycle: creating collections and
ensuring vector search, full-text search, and standard indexes exist.
"""

from dataclasses import dataclass
from functools import lru_cache

from pydantic import ValidationError
from pymongo import ASCENDING, AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.operations import IndexModel, SearchIndexModel

from ..configs.mongo import (
    MongoDBIndexV2Settings,
    get_mongodb_indexv2_settings,
)
from ..embeddings import (
    SupportedEmbeddingModel,
    get_embeddings_service,
)

SEARCH_INDEX_NAME = "search_index"
EXPORT_CONFIG_FILE_INDEX_NAME = "ix_export_config_file"
OPENAI_EMBEDDING_DIMENSIONS = 1536

DEFAULT_TARGET_ID = "DEFAULT"

MONGO_INDEX_V2_NOT_CONFIGURED_MESSAGE = (
    "MongoDB index V2 is not configured for this environment. "
    "Set DEFAULT_INDEX_MONGO_URI, DEFAULT_INDEX_MONGO_DB, and "
    "DEFAULT_INDEX_MONGO_COLLECTION."
)


class MongoIndexV2NotConfiguredError(RuntimeError):
    """Raised when MongoDB index V2 settings are missing required values.

    Callers at the HTTP boundary should translate this into a 503 so the
    response is actionable instead of a generic 500 with a pydantic trace.
    """


@dataclass(frozen=True)
class MongoConnectionTarget:
    """Resolved MongoDB connection coordinates for a single export destination."""

    uri: str
    database: str
    collection: str


@lru_cache
def _get_mongo_client(uri: str) -> AsyncMongoClient:
    """Return a cached ``AsyncMongoClient`` for the given URI."""
    return AsyncMongoClient(uri)


class MongoService:
    """Resolves a target ID to concrete MongoDB connection parameters.

    Accepts an optional ``MongoDBIndexV2Settings`` for testability; when
    omitted the cached singleton from ``get_mongodb_indexv2_settings()``
    is used.
    """

    def __init__(self, settings: MongoDBIndexV2Settings | None = None) -> None:
        self._settings = settings
        self._collection_ensured = False
        self._indexes_ensured = False

    @property
    def settings(self) -> MongoDBIndexV2Settings:
        if self._settings is None:
            try:
                self._settings = get_mongodb_indexv2_settings()
            except ValidationError as e:
                raise MongoIndexV2NotConfiguredError(
                    MONGO_INDEX_V2_NOT_CONFIGURED_MESSAGE
                ) from e
        return self._settings

    def resolve(self, target_id: str) -> MongoConnectionTarget:
        """Return the connection coordinates for *target_id*."""
        return MongoConnectionTarget(
            uri=self.settings.resolve_uri(target_id),
            database=self.settings.resolve_database_name(target_id),
            collection=self.settings.resolve_collection_name(target_id),
        )

    def get_client(self, target_id: str) -> AsyncMongoClient:
        """Return a cached ``AsyncMongoClient`` for *target_id*."""
        return _get_mongo_client(self.resolve(target_id).uri)

    async def get_collection(
        self, target_id: str, embedding_model: SupportedEmbeddingModel | None = None
    ) -> AsyncCollection:
        """Return a ready-to-use ``AsyncCollection`` with indexes ensured."""
        target = self.resolve(target_id)
        client = self.get_client(target_id)
        db = client[target.database]

        if not self._collection_ensured:
            colls = await db.list_collection_names()
            if target.collection not in colls:
                await db.create_collection(target.collection)
            self._collection_ensured = True

        coll = db[target.collection]

        if not self._indexes_ensured:
            await self._ensure_standard_indexes(coll)
            await self._ensure_search_index(coll)
            self._indexes_ensured = True

        return coll

    # ------------------------------------------------------------------
    # Index management (internal)
    # ------------------------------------------------------------------

    @staticmethod
    async def _ensure_standard_indexes(coll: AsyncCollection) -> None:
        """Create standard MongoDB indexes for efficient aggregation queries.

        The compound index on (export_config_id, parsed_directory_file_id)
        supports the $match + $group pipeline used by the export diff to
        list distinct files per export config.
        """
        existing = {idx["name"] async for idx in await coll.list_indexes()}
        if EXPORT_CONFIG_FILE_INDEX_NAME not in existing:
            await coll.create_indexes(
                [
                    IndexModel(
                        [
                            ("export_config_id", ASCENDING),
                            ("parsed_directory_file_id", ASCENDING),
                        ],
                        name=EXPORT_CONFIG_FILE_INDEX_NAME,
                    )
                ]
            )

    @staticmethod
    async def _ensure_search_index(
        coll: AsyncCollection, embedding_model: SupportedEmbeddingModel | None = None
    ) -> None:
        dims = get_embeddings_service().resolve_dims(embedding_model)
        # keep backward compatibility: if the model is default (OpenaAI embeddings),
        # leave the name of the index as the one existing before the possibility
        # of specifying embedding models
        search_idx = SEARCH_INDEX_NAME + (
            "" if not embedding_model else f"-{embedding_model}"
        )
        async for idx in await coll.list_search_indexes():
            if idx.get("name") == search_idx:
                return
        await coll.create_search_index(
            SearchIndexModel(
                name=search_idx,
                type="search",
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "content": {"type": "string"},
                            "_id": {"type": "token"},
                            "tenant_id": {"type": "token"},
                            "export_config_id": {"type": "token"},
                            "parsed_directory_file_id": {"type": "token"},
                            "metadata_fields": {
                                "type": "document",
                                "dynamic": True,
                            },
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": dims,
                                "similarity": "cosine",
                                "quantization": "scalar",
                            },
                        },
                    }
                },
            )
        )


mongo_service = MongoService()


def get_mongo_service() -> MongoService:
    """Return the module-level singleton."""
    return mongo_service
