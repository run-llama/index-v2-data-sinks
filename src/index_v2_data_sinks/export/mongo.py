"""MongoDB export target implementation."""

import functools
from typing import Any

from ..embeddings import (
    SupportedEmbeddingModel,
)
from ..schema import ExportedOutput
from ..services.mongo import (
    DEFAULT_TARGET_ID,
    MongoService,
    get_mongo_service,
)
from ..utils import compute_metadata_hash
from .base import Exporter
from .snapshot import ExportSnapshot


class MongoExporter(Exporter):
    """Writes ExportedOutput documents to MongoDB.

    Combines ``MongoExportTargetService`` (connection/index management)
    with the actual document insertion logic so MongoDB export is fully
    self-contained.

    Uses a delete-then-insert strategy per file so that re-parsed files
    (which may produce a different number of chunks) are fully replaced
    rather than duplicated.
    """

    def __init__(
        self,
        target_service: MongoService | None = None,
        target_id: str = DEFAULT_TARGET_ID,
    ) -> None:
        self._target_service = target_service or get_mongo_service()
        self._target_id = target_id

    async def export(
        self,
        outputs: list[ExportedOutput],
        *,
        project_id: str,
        embeddings: list[list[float]] | None = None,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> None:
        """Replace exported outputs for a file in the target MongoDB collection.

        Deletes any existing chunks for the same
        (parsed_directory_file_id, export_config_id) pair before inserting,
        so re-parsed files with a different chunk count are handled correctly.
        """
        if embeddings is None:
            raise ValueError("`embeddings` must be provided for MongoDB export")

        if not outputs:
            return

        coll = await self._target_service.get_collection(
            self._target_id, embedding_model
        )

        # All outputs in a single export() call belong to the same file+config.
        parsed_directory_file_id = outputs[0].parsed_directory_file_id
        export_config_id = outputs[0].export_config_id

        # Remove previous chunks for this file, then insert the new ones.
        await coll.delete_many(
            {
                "parsed_directory_file_id": parsed_directory_file_id,
                "export_config_id": export_config_id,
            }
        )
        await coll.insert_many(
            [
                o.to_raw_bson_doc_with_embedding(embeddings[i])
                for i, o in enumerate(outputs)
            ]
        )

    async def delete_file(
        self,
        *,
        parsed_directory_file_id: str,
        export_config_id: str,
    ) -> int:
        """Delete all chunks for a file from MongoDB.

        Returns the number of documents deleted.
        """
        coll = await self._target_service.get_collection(self._target_id)
        result = await coll.delete_many(
            {
                "parsed_directory_file_id": parsed_directory_file_id,
                "export_config_id": export_config_id,
            }
        )
        return result.deleted_count

    async def delete_files(
        self,
        *,
        parsed_directory_file_ids: list[str],
        export_config_id: str,
    ) -> int:
        """Delete all chunks for multiple files in a single query.

        Returns the total number of documents deleted.
        """
        if not parsed_directory_file_ids:
            return 0
        coll = await self._target_service.get_collection(self._target_id)
        result = await coll.delete_many(
            {
                "parsed_directory_file_id": {"$in": parsed_directory_file_ids},
                "export_config_id": export_config_id,
            }
        )
        return result.deleted_count

    async def list_snapshots(
        self,
        *,
        export_config_id: str,
    ) -> list[ExportSnapshot]:
        """Return one snapshot per file currently in MongoDB for the given export config.

        Uses an aggregation pipeline that groups chunks by
        ``parsed_directory_file_id`` and captures a metadata hash from
        the first chunk. Requires a compound index on
        ``(export_config_id, parsed_directory_file_id)`` for efficiency.
        """
        coll = await self._target_service.get_collection(self._target_id)
        pipeline: list[dict[str, Any]] = [
            {"$match": {"export_config_id": export_config_id}},
            {
                "$group": {
                    "_id": "$parsed_directory_file_id",
                    "metadata_fields": {"$first": "$metadata_fields"},
                    "content_fingerprint_hash": {"$first": "$content_fingerprint_hash"},
                    "source_modified_at": {"$first": "$source_modified_at"},
                    "exported_at": {"$first": "$exported_at"},
                }
            },
        ]
        snapshots: list[ExportSnapshot] = []
        async for doc in await coll.aggregate(pipeline):
            raw_meta = doc.get("metadata_fields")
            meta_hash = compute_metadata_hash(raw_meta) if raw_meta else None
            snapshots.append(
                ExportSnapshot(
                    parsed_directory_file_id=doc["_id"],
                    metadata_fields_hash=meta_hash,
                    content_fingerprint_hash=doc.get("content_fingerprint_hash"),
                    source_modified_at=doc.get("source_modified_at"),
                    exported_at=doc.get("exported_at"),
                )
            )
        return snapshots


mongo_exporter = MongoExporter()


@functools.lru_cache(maxsize=1)
def get_mongo_exporter() -> MongoExporter:
    return mongo_exporter
