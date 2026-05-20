import asyncio
from asyncio import Semaphore

from index_v2_data_sinks.embeddings import SupportedEmbeddingModel, embeddings_service
from index_v2_data_sinks.export.mongo import get_mongo_exporter

from .chunking import ChunkerService, ChunkingStrategy
from .parsed_outputs import get_parsed_outputs
from .schema import (
    ExportedOutputV1,
    MetadataDict,
    ParseSyncOutput,
    _build_page_boundaries,
)

MAX_CONCURRENT_EXPORTS = 10


async def mongo_export_pipeline(
    directory_name: str,
    project_id: str,
    config_id: str,
    chunking_strategy: ChunkingStrategy = "sentence",
    metadata: MetadataDict | None = None,
    embedding_model: SupportedEmbeddingModel | None = None,
) -> None:
    mongo_exporter = get_mongo_exporter()
    parsed_outputs, directory_id = await get_parsed_outputs(directory_name)
    semaphore = Semaphore(MAX_CONCURRENT_EXPORTS)

    async def export(po: ParseSyncOutput) -> None:
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
            await mongo_exporter.export(
                exported_outputs, embeddings=embeddings, embedding_model=embedding_model
            )

    await asyncio.gather(*[export(po) for po in parsed_outputs])
