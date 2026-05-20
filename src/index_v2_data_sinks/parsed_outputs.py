import asyncio
import functools
import os
from asyncio import Semaphore

import httpx
from llama_cloud import AsyncLlamaCloud

from index_v2_data_sinks.schema import ParseSyncOutput, ParseSyncOutputV1

MAX_DOWNLOADS = 50


@functools.lru_cache(maxsize=1)
def get_lc_client() -> AsyncLlamaCloud:
    return AsyncLlamaCloud(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
    )


async def _file_to_parsed_output(file_id: str) -> ParseSyncOutput:
    client = get_lc_client()
    presigned_url = await client.files.get(file_id)
    async with httpx.AsyncClient() as client:
        response = await client.get(presigned_url.url)
        response.raise_for_status()
        return ParseSyncOutputV1.model_validate_json(response.content)


async def get_parsed_outputs(directory_name: str) -> tuple[list[ParseSyncOutput], str]:
    client = get_lc_client()
    response = await client.beta.directories.list(
        name=directory_name,
        page_size=1,
        type="index",
    )
    directory_id = response.items[0].id
    page_token = None
    files = []
    while True:
        files_resp = await client.beta.directories.files.list(
            directory_id=directory_id, page_token=page_token
        )
        files.extend([f.id for f in files_resp.items])
        if files_resp.next_page_token is None:
            break
        page_token = files_resp.next_page_token
    semaphore = Semaphore()

    async def get_file_sem(file_id: str):
        async with semaphore:
            return await _file_to_parsed_output(file_id)

    return await asyncio.gather(*[get_file_sem(fid) for fid in files]), directory_id
