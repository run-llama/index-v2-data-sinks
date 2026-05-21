import asyncio
from asyncio import Semaphore
from typing import Literal

from .chunking import ChunkerService, ChunkingStrategy
from .embeddings import SupportedEmbeddingModel, embeddings_service
from .export.mongo import get_mongo_exporter
from .export.pgvector import get_pgvector_exporter
from .parsed_outputs import get_parsed_outputs
from .schema import (
    ExportedOutputV1,
    MetadataDict,
    ParseSyncOutput,
    _build_page_boundaries,
)

MAX_CONCURRENT_EXPORTS = 10


async def export_pipeline(
    export_to: Literal["mongo", "postgres"],
    directory_name: str,
    project_id: str,
    config_id: str,
    chunking_strategy: ChunkingStrategy = "sentence",
    metadata: MetadataDict | None = None,
    embedding_model: SupportedEmbeddingModel | None = None,
) -> None:
    if export_to == "mongo":
        exporter = get_mongo_exporter()
    else:
        exporter = get_pgvector_exporter()
    parsed_outputs, directory_id = await get_parsed_outputs(directory_name, project_id)
    print(f"Received {len(parsed_outputs)} parsed outputs")
    semaphore = Semaphore(MAX_CONCURRENT_EXPORTS)

    async def export(po: ParseSyncOutput, i: int) -> None:
        async with semaphore:
            _, segments = _build_page_boundaries(po)
            chunks = ChunkerService().chunk("\n".join(segments), chunking_strategy)
            embeddings = await embeddings_service.embed(
                [c.text for c in chunks], embedding_model
            )
            exported_outputs = []
            for chunk in chunks:
                e = ExportedOutputV1.build(
                    po, project_id, config_id, chunk, directory_id, metadata=metadata
                )
                exported_outputs.append(e)
            await exporter.export(
                exported_outputs, embeddings=embeddings, embedding_model=embedding_model
            )
            print(f"Exported {i + 1} / {len(parsed_outputs)} files")

    await asyncio.gather(*[export(po, i) for (i, po) in enumerate(parsed_outputs)])
