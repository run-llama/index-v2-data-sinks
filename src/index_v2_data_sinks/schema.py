from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypeAlias, TypeGuard

from bson import encode
from bson.raw_bson import RawBSONDocument
from pydantic import BaseModel, Field

MetadataScalarValue: TypeAlias = str | int | float | bool | None
MetadataListValue: TypeAlias = list[str]
MetadataValue: TypeAlias = MetadataScalarValue | MetadataListValue
MetadataDict: TypeAlias = dict[str, MetadataValue]
JSONType = dict[str, Any] | list[Any] | str | int | float | bool | None

# Convenience alias for dicts whose values are generic JSON types.
JSONDictType = dict[str, JSONType]


class PageBoundary(BaseModel):
    """Maps a page number to its character offset range in the concatenated text."""

    page_number: int = Field(description="1-based page number.")
    start_char: int = Field(
        description="Start character offset in concatenated text (inclusive)."
    )
    end_char: int = Field(
        description="End character offset in concatenated text (exclusive)."
    )


class AttachmentRef(BaseModel):
    """Reference to a file attachment, retrievable via ``GET /api/v1/beta/attachments/{attachment_name}?source_id=...``."""

    model_config = {"extra": "ignore"}

    type: str = Field(description="Attachment kind, e.g. 'screenshot', 'items'.")
    attachment_name: str = Field(
        description="Attachment-relative path, e.g. 'screenshots/page_7.jpg'."
    )
    source_id: str = Field(
        description="File ID to pass as source_id when fetching the attachment."
    )


class ParsedPage(BaseModel):
    """A single page of parsed markdown content from LlamaParse."""

    model_config = {"extra": "ignore"}

    page_number: int = Field(description="1-based page number.")
    text: str | None = Field(default=None, description="Parsed text content.")
    header: str | None = Field(default=None, description="Extracted header text.")
    footer: str | None = Field(default=None, description="Extracted footer text.")
    attachments: dict[str, AttachmentRef] = Field(
        default_factory=dict,
        description="Attachment references keyed by kind (e.g. 'screenshot', 'items').",
    )


class ParseSyncOutputV1(BaseModel):
    """V1 of the directory parse sync output format.

    Stores the page-by-page markdown from LlamaParse OutputFilesV2.markdown.
    Provenance metadata (parse_job_id, source_file_id, etc.) is tracked on
    the output DirectoryFile's internal_metadata, not duplicated here.
    """

    model_config = {"extra": "ignore"}

    version: Literal["v1"] = Field("v1", description="Schema version discriminator.")
    pages: list[ParsedPage] = Field(description="Page-by-page parsed markdown content.")


ParseSyncOutput = Annotated[
    ParseSyncOutputV1,
    Field(discriminator="version"),
]


class Chunk(BaseModel):
    """A single chunk produced by the chunk-maker service."""

    index: int = Field(description="0-based position of this chunk in the sequence.")
    text: str = Field(description="The chunk text content.")
    start_char_idx: int | None = Field(
        default=None, description="Start character offset in the original text."
    )
    end_char_idx: int | None = Field(
        default=None, description="End character offset in the original text."
    )
    token_count: int | None = Field(
        default=None, description="Approximate token count for this chunk."
    )


class ExportedOutputV1(BaseModel):
    """
    v1 of the class representing an exported ParseSyncOutput.
    """

    version: Literal["v1"] = Field(default="v1")
    content: str = Field(description="Content of the chunk")
    chunk_index: int = Field(description="0-based index for chunking")
    chunk_start_char: int | None = Field(
        description="Index of the starting character for the chunk"
    )
    chunk_end_char: int | None = Field(
        description="Index of the ending character for the chunk"
    )
    chunk_token_count: int | None = Field(
        description="Token count for the specific character"
    )
    parsed_directory_file_id: str = Field(
        description="ID of the file storing the ParseSyncOutput JSON data"
    )
    source_file_id: str | None = Field(
        default=None,
        description="File storage ID of the original source file that was parsed.",
    )
    page_range_start: int = Field(description="First page covered by the content")
    page_range_end: int = Field(description="Last page covered by the content")
    tenant_id: str = Field(
        description="Tenant ID (for tenant detection, corresponding to the platform's project ID)"
    )
    export_config_id: str = Field(
        description="ID of the directory sync configuration used to produce the file"
    )
    metadata_fields: MetadataDict | None = None
    content_fingerprint_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the source file's content_fingerprint, for content change detection.",
    )
    source_modified_at: datetime | None = Field(
        default=None,
        description=(
            "Source-side modified_at at the time this chunk was exported, used as a "
            "timestamp signal for content change detection on the next export."
        ),
    )
    exported_at: datetime | None = Field(
        default=None,
        description="Timestamp when this chunk was exported, for timestamp-based change detection.",
    )
    attachments: list[AttachmentRef] = Field(
        default_factory=list, description="Attachments related to the exported output"
    )

    @classmethod
    def build(
        cls,
        output: ParseSyncOutput,
        project_id: str,
        config_id: str,
        chunk: Chunk,
        parsed_directory_file_id: str,
        metadata: MetadataDict | None = None,
        source_file_id: str | None = None,
        content_fingerprint_hash: str | None = None,
        source_modified_at: datetime | None = None,
    ) -> "ExportedOutputV1":
        if output.version != "v1":
            raise TypeError(
                f"Unsupported output version: {output.version}"
            )  # code is unreachable for now, but saves us from silent failures if we change the version in the future
        if len(output.pages) == 0:
            raise ValueError(
                "Cannot construct the exported output with an empty sequence of pages"
            )
        # isolate only the pages concerning the speicifc chunk
        boundaries, _ = _build_page_boundaries(output)
        overlapping = _overlapping_pages(chunk, boundaries)
        overlapping = sorted(overlapping, key=lambda x: x.page_number)
        page_range_start = overlapping[0].page_number
        page_range_end = overlapping[-1].page_number
        attachments = [
            ref
            for page in output.pages
            if page_range_start <= page.page_number <= page_range_end
            for ref in page.attachments.values()
        ]

        return cls(
            content=chunk.text,
            chunk_token_count=chunk.token_count,
            chunk_start_char=chunk.start_char_idx,
            chunk_end_char=chunk.end_char_idx,
            chunk_index=chunk.index,
            parsed_directory_file_id=parsed_directory_file_id,
            source_file_id=source_file_id,
            page_range_start=page_range_start,
            page_range_end=page_range_end,
            tenant_id=project_id,
            metadata_fields=metadata,
            content_fingerprint_hash=content_fingerprint_hash,
            source_modified_at=source_modified_at,
            exported_at=datetime.now(UTC),
            export_config_id=config_id,
            attachments=attachments,
        )

    def to_raw_bson_doc_with_embedding(self, embedding: list[float]) -> RawBSONDocument:
        doc = self.model_dump()
        doc["embedding"] = embedding
        return RawBSONDocument(encode(doc))


ExportedOutput = Annotated[
    ExportedOutputV1,
    Field(discriminator="version"),
]


class ExtractSyncOutputV1(BaseModel):
    """V1 of the directory extract sync output format.

    Stores the structured extraction result from LlamaExtract.
    Provenance metadata is tracked on the output DirectoryFile's internal_metadata.
    """

    model_config = {"extra": "ignore"}

    version: Literal["v1"] = Field("v1", description="Schema version discriminator.")
    product_type: Literal["extract"] = Field(
        "extract", description="Discriminator vs parse output."
    )
    extract_result: JSONDictType | list[JSONDictType] = Field(
        description="Structured extraction result."
    )
    extraction_target: str = Field(
        description="Extraction target used (e.g. per_doc, per_page)."
    )
    data_schema: JSONDictType = Field(description="JSON schema used for extraction.")
    extract_metadata: JSONDictType | None = Field(
        default=None, description="Additional extraction metadata."
    )


ExtractSyncOutput = Annotated[
    ExtractSyncOutputV1,
    Field(discriminator="version"),
]


class DirectorySyncOutputV1(BaseModel):
    """Unified envelope for all sync output types.

    A single blob per source file wraps both parse and extract results.
    """

    version: Literal["v1"] = Field("v1", description="Schema version discriminator.")
    parse: ParseSyncOutputV1 | None = Field(
        default=None, description="Parse output, if parse was run."
    )
    extract: ExtractSyncOutput | None = Field(
        default=None, description="Extract output, if extract was run."
    )


DirectorySyncOutput = Annotated[
    DirectorySyncOutputV1,
    Field(discriminator="version"),
]


def is_metadata_value(value: object) -> TypeGuard[MetadataValue]:
    """Return whether a value fits the directory metadata contract."""
    if value is None or isinstance(value, str | int | float | bool):
        return True
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _build_page_boundaries(
    output: ParseSyncOutputV1,
) -> tuple[list[PageBoundary], list[str]]:
    page_boundaries: list[PageBoundary] = []
    segments = []
    running_offset = 0

    for page in output.pages:
        page_content = "\n".join(filter(None, [page.header, page.text, page.footer]))
        if not page_content:
            continue
        segments.append(page_content)
        start = running_offset
        # Account for the "\n" separator between segments
        if running_offset > 0:
            running_offset += 1  # newline separator
        running_offset += len(page_content)
        page_boundaries.append(
            PageBoundary(
                page_number=page.page_number,
                start_char=start + (1 if start > 0 else 0),
                end_char=running_offset,
            )
        )
    if not page_boundaries:
        raise ValueError("No pages with text content were found")
    return page_boundaries, segments


def _overlapping_pages(
    chunk: Chunk,
    page_boundaries: list[PageBoundary],
) -> list[PageBoundary]:
    """Return page boundaries that overlap with the chunk's character range."""
    if chunk.start_char_idx is None or chunk.end_char_idx is None:
        return page_boundaries
    return [
        pb
        for pb in page_boundaries
        if pb.start_char < chunk.end_char_idx and pb.end_char > chunk.start_char_idx
    ]
