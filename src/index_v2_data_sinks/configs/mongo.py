from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MongoDBIndexV2Settings(BaseSettings):
    """Settings for MongoDB in Index V2"""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_by_name=True,
        env_prefix="DEFAULT_INDEX_MONGO_",
    )

    # Name of the database in MongoDB
    db: str = Field(
        description="Name of the MongoDB database",
    )

    # URI for MongoDB, necessary for connection
    uri: str = Field(
        description="Connection URI for MongoDB",
        json_schema_extra={"is_secret": True},
    )

    # Collection name within the MongoDB database
    collection: str = Field(
        description="Collection name to use within the MongoDB database"
    )

    uri_map: dict[str, str] = Field(
        description="Map for MongoDB URLs",
        default_factory=lambda: {},
        json_schema_extra={"is_secret": True},
    )  # make mypy happy instead of using dict as factory
    db_map: dict[str, str] = Field(
        description="Map for MongoDB database names", default_factory=lambda: {}
    )  # make mypy happy instead of using dict as factory
    collections_map: dict[str, str] = Field(
        description="Map for MongoDB database collections", default_factory=lambda: {}
    )  # make mypy happy instead of using dict as factory

    def get_redacted_uri(self) -> str:
        prefix = (
            "mongodb+srv://" if self.uri.startswith("mongodb+srv://") else "mongodb://"
        )
        to_redact = self.uri.removeprefix(prefix)
        return prefix + "*" * len(to_redact)

    def resolve_uri(self, k: str) -> str:
        """
        Resolve the URI based on its namespace (as a key of the URI map we have in the settings)
        """
        return self.uri_map.get(k, self.uri)

    def resolve_database_name(self, k: str) -> str:
        """
        Resolve the database name based on its namespace (as a key of the database name map we have in the settings)
        """
        return self.db_map.get(k, self.db)

    def resolve_collection_name(self, k: str) -> str:
        """
        Resolve the collection name based on its namespace (as a key of the collection name map we have in the settings)
        """
        return self.collections_map.get(k, self.collection)


@lru_cache
def get_mongodb_indexv2_settings() -> MongoDBIndexV2Settings:
    """Get cached MongoDB IndexV2 settings."""
    # fields are loaded from env variables, ignore call issues reports
    return MongoDBIndexV2Settings()  # type: ignore
