"""Shared snapshot schema for export targets that support diff-based exports.

A snapshot is the lightweight per-file record an export target reports back
so the orchestrator can diff it against the current sync directory state.
The schema is target-agnostic: each target translates its own storage layout
into this form.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ExportSnapshot(BaseModel):
    """One file currently present in an export target for a given config."""

    parsed_directory_file_id: str = Field(
        description="The DirectoryFile ID in the output (sync) directory."
    )
    metadata_fields_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the metadata stored alongside the file, or None if no metadata.",
    )
    content_fingerprint_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the source file's content_fingerprint, or None for legacy docs.",
    )
    source_modified_at: datetime | None = Field(
        default=None,
        description=(
            "Source-side modified_at recorded when the file was last exported. "
            "Preferred timestamp signal for change detection; None for legacy docs."
        ),
    )
    exported_at: datetime | None = Field(
        default=None,
        description="Timestamp when the file was last exported, or None for legacy docs.",
    )

    def diff_key(self) -> str:
        return self.parsed_directory_file_id
