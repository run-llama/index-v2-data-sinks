"""Common protocol for export target implementations.

Each export target (MongoDB, PGVector, ...) implements the same async
surface so callers can swap targets without conditional logic. The
protocol is ``runtime_checkable`` so it can be used in ``isinstance``
checks where needed.
"""

from typing import Protocol, runtime_checkable

from ..embeddings import SupportedEmbeddingModel
from ..schema import ExportedOutput
from .snapshot import ExportSnapshot


@runtime_checkable
class Exporter(Protocol):
    """Async protocol every export target implementation must satisfy."""

    async def export(
        self,
        outputs: list[ExportedOutput],
        *,
        embeddings: list[list[float]] | None = None,
        embedding_model: SupportedEmbeddingModel | None = None,
    ) -> None:
        """Replace the exported outputs for a single (file, export_config) pair.

        All ``outputs`` in one call must belong to the same
        ``parsed_directory_file_id`` and ``export_config_id``. Implementations
        should delete existing rows/documents for that pair before inserting
        the new ones so re-parsed files with a different chunk count are
        handled correctly.
        """
        ...

    async def delete_file(
        self,
        *,
        parsed_directory_file_id: str,
        export_config_id: str,
    ) -> int:
        """Delete all chunks for a single file. Returns the number deleted."""
        ...

    async def delete_files(
        self,
        *,
        parsed_directory_file_ids: list[str],
        export_config_id: str,
    ) -> int:
        """Delete all chunks for multiple files. Returns the total number deleted."""
        ...

    async def list_snapshots(
        self,
        *,
        export_config_id: str,
    ) -> list[ExportSnapshot]:
        """Return one snapshot per file currently in the target for the given config."""
        ...
