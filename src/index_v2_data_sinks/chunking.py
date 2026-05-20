import functools
from typing import Literal

from chonkie import FastChunker, SentenceChunker, TokenChunker
from typing_extensions import overload

from .schema import Chunk

ChunkingStrategy = Literal["fast", "sentence", "token"]


@functools.lru_cache(maxsize=1)
def _get_fast_chunker() -> FastChunker:
    return FastChunker()


@functools.lru_cache(maxsize=1)
def _get_sentence_chunker() -> SentenceChunker:
    return SentenceChunker()


@functools.lru_cache(maxsize=1)
def _get_token_chunker() -> TokenChunker:
    return TokenChunker()


class ChunkerService:
    @overload
    def chunk(self, input: str, strategy: ChunkingStrategy) -> list[Chunk]: ...

    @overload
    def chunk(
        self, input: list[str], strategy: ChunkingStrategy
    ) -> list[list[Chunk]]: ...

    def chunk(
        self, input: str | list[str], strategy: ChunkingStrategy
    ) -> list[Chunk] | list[list[Chunk]]:
        if strategy == "fast":
            chunker = _get_fast_chunker()
        elif strategy == "sentence":
            chunker = _get_sentence_chunker()
        else:
            chunker = _get_sentence_chunker()
        if isinstance(input, str):
            chunks = chunker.chunk(input)
            actual = [
                Chunk(
                    index=i,
                    text=c.text,
                    token_count=c.token_count,
                    start_char_idx=c.start_index,
                    end_char_idx=c.end_index,
                )
                for i, c in enumerate(chunks)
            ]
        else:
            batches = chunker.chunk_batch(input)
            actual = []
            for batch in batches:
                a = [
                    Chunk(
                        index=i,
                        text=c.text,
                        token_count=c.token_count,
                        start_char_idx=c.start_index,
                        end_char_idx=c.end_index,
                    )
                    for i, c in enumerate(batch)
                ]
                actual.append(a)
        return actual
