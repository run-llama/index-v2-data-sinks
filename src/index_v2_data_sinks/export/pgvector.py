"""PGVector export target implementation."""

import functools
import uuid
from typing import Any

from sqlalchemy import delete, func, select

from ..embeddings import (
    SupportedEmbeddingModel,
)
from ..schema import ExportedOutput
from ..services.pgvector import (
    DEFAULT_TARGET_ID,
    PGVectorService,
    get_pgvector_service,
)
from ..utils import compute_metadata_hash
from .base import Exporter
from .snapshot import ExportSnapshot


class PGVectorExporter(Exporter):
    """Writes ExportedOutput rows to a PostgreSQL table with a pgvector column.

    Combines ``PGVectorService`` (connection/schema/index management) with
    the actual row insertion logic so PGVector export is fully
    self-contained.

    Uses a delete-then-insert strategy per file so that re-parsed files
    (which may produce a different number of chunks) are fully replaced
    rather than duplicated.
    """

    def __init__(
        self,
        target_service: PGVectorService | None = None,
        target_id: str = DEFAULT_TARGET_ID,
    ) -> None:
        self._target_service = target_service or get_pgvector_service()
        self._target_id = target_id

    async def export(
        self,
        outputs: list[ExportedOutput],
        *,
        embeddings: list[list[float]] | None = None,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> None:
        """Replace exported outputs for a file in the target PGVector table.

        Deletes any existing chunks for the same
        (parsed_directory_file_id, export_config_id) pair before inserting,
        so re-parsed files with a different chunk count are handled correctly.
        """
        if embeddings is None:
            raise ValueError("`embeddings` must be provided for PGVector export")

        if not outputs:
            return

        engine, table = await self._target_service.get_table(
            self._target_id, embedding_model
        )

        # All outputs in a single export() call belong to the same file+config.
        parsed_directory_file_id = outputs[0].parsed_directory_file_id
        export_config_id = outputs[0].export_config_id

        rows = [
            self._build_row(o, embeddings[i]) for i, o in enumerate(outputs)
        ]

        async with engine.begin() as conn:
            await conn.execute(
                delete(table).where(
                    table.c.parsed_directory_file_id == parsed_directory_file_id,
                    table.c.export_config_id == export_config_id,
                )
            )
            await conn.execute(table.insert(), rows)

    @staticmethod
    def _build_row(output: ExportedOutput, embedding: list[float]) -> dict[str, Any]:
        data = output.model_dump(mode="json")
        # We need to preserve the actual datetime objects on these columns
        # (model_dump(mode="json") stringifies them), so re-pull from the
        # model directly.
        data["source_modified_at"] = output.source_modified_at
        data["exported_at"] = output.exported_at
        data.pop("version", None)
        data["id"] = str(uuid.uuid4())
        data["embedding"] = embedding
        return data

    async def delete_file(
        self,
        *,
        parsed_directory_file_id: str,
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> int:
        """Delete all chunks for a file from PGVector.

        Returns the number of rows deleted.
        """
        engine, table = await self._target_service.get_table(
            self._target_id, embedding_model
        )
        async with engine.begin() as conn:
            result = await conn.execute(
                delete(table).where(
                    table.c.parsed_directory_file_id == parsed_directory_file_id,
                    table.c.export_config_id == export_config_id,
                )
            )
        return result.rowcount or 0

    async def delete_files(
        self,
        *,
        parsed_directory_file_ids: list[str],
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> int:
        """Delete all chunks for multiple files in a single statement.

        Returns the total number of rows deleted.
        """
        if not parsed_directory_file_ids:
            return 0
        engine, table = await self._target_service.get_table(
            self._target_id, embedding_model
        )
        async with engine.begin() as conn:
            result = await conn.execute(
                delete(table).where(
                    table.c.parsed_directory_file_id.in_(parsed_directory_file_ids),
                    table.c.export_config_id == export_config_id,
                )
            )
        return result.rowcount or 0

    async def list_snapshots(
        self,
        *,
        export_config_id: str,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> list[ExportSnapshot]:
        """Return one snapshot per file currently in PGVector for the given export config.

        Picks an arbitrary row per ``parsed_directory_file_id`` (via MIN on
        chunk_index-equivalent columns) to source the metadata / fingerprint
        / timestamps. The composite index on
        ``(export_config_id, parsed_directory_file_id)`` keeps this efficient.
        """
        engine, table = await self._target_service.get_table(
            self._target_id, embedding_model
        )
        stmt = (
            select(
                table.c.parsed_directory_file_id,
                func.min(table.c.chunk_index).label("min_chunk_index"),
            )
            .where(table.c.export_config_id == export_config_id)
            .group_by(table.c.parsed_directory_file_id)
        )

        snapshots: list[ExportSnapshot] = []
        async with engine.connect() as conn:
            grouped = (await conn.execute(stmt)).all()
            if not grouped:
                return []

            # Fetch the representative row (chunk_index == min) for each file.
            for parsed_directory_file_id, min_chunk_index in grouped:
                row_stmt = select(
                    table.c.metadata_fields,
                    table.c.content_fingerprint_hash,
                    table.c.source_modified_at,
                    table.c.exported_at,
                ).where(
                    table.c.export_config_id == export_config_id,
                    table.c.parsed_directory_file_id == parsed_directory_file_id,
                    table.c.chunk_index == min_chunk_index,
                ).limit(1)
                row = (await conn.execute(row_stmt)).first()
                if row is None:
                    continue
                raw_meta = row.metadata_fields
                meta_hash = compute_metadata_hash(raw_meta) if raw_meta else None
                snapshots.append(
                    ExportSnapshot(
                        parsed_directory_file_id=parsed_directory_file_id,
                        metadata_fields_hash=meta_hash,
                        content_fingerprint_hash=row.content_fingerprint_hash,
                        source_modified_at=row.source_modified_at,
                        exported_at=row.exported_at,
                    )
                )
        return snapshots


pgvector_exporter = PGVectorExporter()


@functools.lru_cache(maxsize=1)
def get_pgvector_exporter() -> PGVectorExporter:
    return pgvector_exporter
