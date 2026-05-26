"""Tests for index_v2_data_sinks.pipeline.export_pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from index_v2_data_sinks import pipeline as pipeline_mod
from index_v2_data_sinks.schema import ParsedPage, ParseSyncOutputV1


def _make_output(
    text: str = "hello world. another sentence here.",
) -> ParseSyncOutputV1:
    return ParseSyncOutputV1(version="v1", pages=[ParsedPage(page_number=1, text=text)])


async def test_export_pipeline_calls_exporter_for_each_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = [_make_output("first doc text here."), _make_output("second doc text.")]

    monkeypatch.setattr(
        pipeline_mod,
        "get_parsed_outputs",
        AsyncMock(return_value=(outputs, "dir-id")),
    )

    # Stub embeddings so we don't need API keys / network.
    fake_embeddings_service = MagicMock()
    fake_embeddings_service.embed = AsyncMock(
        side_effect=lambda chunks, model: [[0.1] for _ in chunks]
    )
    monkeypatch.setattr(pipeline_mod, "embeddings_service", fake_embeddings_service)

    # Stub the exporter factory so no real backend client is constructed.
    exporter = MagicMock()
    exporter.export = AsyncMock()
    factories = {**pipeline_mod.EXPORT_TARGET_TO_FACTORY, "mongo": lambda: exporter}
    monkeypatch.setattr(pipeline_mod, "EXPORT_TARGET_TO_FACTORY", factories)

    await pipeline_mod.export_pipeline(
        export_to="mongo",
        directory_name="my-dir",
        project_id="proj-1",
        config_id="cfg-1",
    )

    # One export() call per parsed output.
    assert exporter.export.await_count == len(outputs)
    for call in exporter.export.await_args_list:
        kwargs = call.kwargs
        # Each call exports a list of ExportedOutputV1 with matching embeddings.
        exported_outputs = call.args[0]
        assert len(exported_outputs) == len(kwargs["embeddings"])
        assert kwargs["project_id"] == "proj-1"


async def test_export_pipeline_unknown_target_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_mod,
        "get_parsed_outputs",
        AsyncMock(return_value=([_make_output()], "dir-id")),
    )
    with pytest.raises(KeyError):
        await pipeline_mod.export_pipeline(
            export_to="does-not-exist",  # type: ignore[arg-type]
            directory_name="d",
            project_id="p",
            config_id="c",
        )
