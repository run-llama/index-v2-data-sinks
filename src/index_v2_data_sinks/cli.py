import asyncio
import json
from typing import Annotated

import typer
from typer import Typer

from .chunking import ChunkingStrategy
from .embeddings import SupportedEmbeddingModel
from .pipeline import ExportTarget, export_pipeline

app = Typer(help="Index v2 data sinks CLI.")


@app.command("export")
def export(
    directory_name: Annotated[
        str, typer.Option("--directory-name", "-d", help="LlamaCloud directory name.")
    ],
    project_id: Annotated[
        str, typer.Option("--project-id", "-p", help="LlamaCloud project ID.")
    ],
    config_id: Annotated[
        str, typer.Option("--config-id", "-c", help="Export config ID.")
    ],
    export_to: Annotated[
        ExportTarget,
        typer.Option(
            "--export-to",
            "-t",
            help="Target sink to export to.",
            case_sensitive=False,
        ),
    ] = "mongo",
    chunking_strategy: Annotated[
        ChunkingStrategy,
        typer.Option(
            "--chunking-strategy",
            "-s",
            help="Chunking strategy: fast | sentence | token.",
        ),
    ] = "sentence",
    embedding_model: Annotated[
        SupportedEmbeddingModel | None,
        typer.Option(
            "--embedding-model",
            "-e",
            help="Embedding model identifier (e.g. openai:text-embedding-3-small).",
        ),
    ] = None,
    metadata: Annotated[
        str | None,
        typer.Option(
            "--metadata",
            "-m",
            help="Optional JSON-encoded metadata dict attached to each chunk.",
        ),
    ] = None,
) -> None:
    """Run the export pipeline for a LlamaCloud directory."""
    parsed_metadata = None
    if metadata is not None:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"--metadata must be valid JSON: {e}") from e
        if not isinstance(parsed_metadata, dict):
            raise typer.BadParameter("--metadata must decode to a JSON object.")

    asyncio.run(
        export_pipeline(
            export_to=export_to,  # type: ignore[arg-type]
            directory_name=directory_name,
            project_id=project_id,
            config_id=config_id,
            chunking_strategy=chunking_strategy,  # type: ignore[arg-type]
            metadata=parsed_metadata,
            embedding_model=embedding_model,  # type: ignore[arg-type]
        )
    )


if __name__ == "__main__":
    app()
