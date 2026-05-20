import functools
import os
from typing import Literal, TypeAlias

from catsu import Client
from typing_extensions import overload

SupportedEmbeddingModel: TypeAlias = Literal[
    "mixedbread:mxbai-embed-large-v1",
    "mixedbread:mxbai-embed-2d-large-v1",
    "mixedbread:deepset-mxbai-embed-de-large-v1",
    "mixedbread:mxbai-embed-xsmall-v1",
    "openai:text-embedding-3-large",
    "openai:text-embedding-3-small",
    "openai:text-embedding-ada-002",
    "togetherai:BAAI/bge-base-en-v1.5",
    "togetherai:BAAI/bge-large-en-v1.5",
    "togetherai:BAAI/bge-m3",
    "togetherai:WhereIsAI/UAE-Large-V1",
    "togetherai:togethercomputer/m2-bert-80M-8k-retrieval",
    "togetherai:Alibaba-NLP/gte-Qwen2-7B-instruct",
    "togetherai:Alibaba-NLP/gte-modernbert-base",
    "deepinfra:Qwen/Qwen3-Embedding-8B",
    "deepinfra:Qwen/Qwen3-Embedding-4B",
    "deepinfra:Qwen/Qwen3-Embedding-0.6B",
    "deepinfra:BAAI/bge-m3",
    "deepinfra:BAAI/bge-base-en-v1.5",
    "deepinfra:BAAI/bge-large-en-v1.5",
    "deepinfra:BAAI/bge-en-icl",
    "deepinfra:intfloat/e5-base-v2",
    "deepinfra:intfloat/e5-large-v2",
    "deepinfra:intfloat/multilingual-e5-large",
    "deepinfra:sentence-transformers/all-MiniLM-L6-v2",
    "deepinfra:sentence-transformers/all-MiniLM-L12-v2",
    "deepinfra:sentence-transformers/all-mpnet-base-v2",
    "deepinfra:thenlper/gte-base",
    "deepinfra:thenlper/gte-large",
    "deepinfra:google/embeddinggemma-300m",
    "gemini:gemini-embedding-001",
    "cloudflare:@cf/baai/bge-small-en-v1.5",
    "cloudflare:@cf/baai/bge-base-en-v1.5",
    "cloudflare:@cf/baai/bge-large-en-v1.5",
    "cloudflare:@cf/baai/bge-m3",
    "cloudflare:@cf/qwen/qwen3-embedding-0.6b",
    "cloudflare:@cf/google/embeddinggemma-300m",
    "cloudflare:@cf/pfnet/plamo-embedding-1b",
    "cohere:embed-v4.0",
    "cohere:embed-english-v3.0",
    "cohere:embed-english-light-v3.0",
    "cohere:embed-multilingual-v3.0",
    "cohere:embed-multilingual-light-v3.0",
    "jinaai:jina-embeddings-v4",
    "jinaai:jina-embeddings-v3",
    "jinaai:jina-code-embeddings-1.5b",
    "jinaai:jina-code-embeddings-0.5b",
    "jinaai:jina-embeddings-v2-base-en",
    "jinaai:jina-embeddings-v2-base-code",
    "voyageai:voyage-4-large",
    "voyageai:voyage-4",
    "voyageai:voyage-4-lite",
    "voyageai:voyage-4-nano",
    "voyageai:voyage-3-large",
    "voyageai:voyage-3.5",
    "voyageai:voyage-3.5-lite",
    "voyageai:voyage-3",
    "voyageai:voyage-3-lite",
    "voyageai:voyage-code-3",
    "voyageai:voyage-finance-2",
    "voyageai:voyage-law-2",
    "voyageai:voyage-code-2",
    "voyageai:voyage-multilingual-2",
    "voyageai:voyage-multimodal-3",
    "mistral:codestral-embed-2505",
    "mistral:mistral-embed",
    "nomic:nomic-embed-text-v1.5",
    "nomic:nomic-embed-text-v1",
]
SupportedEmbeddingProvider: TypeAlias = Literal[
    "nomic",
    "mistral",
    "openai",
    "mixedbread",
    "cohere",
    "voyageai",
    "togetherai",
    "jinaai",
    "gemini",
    "deepinfra",
    "cloudflare",
]

DEFAULT_EMBEDDING_MODEL: SupportedEmbeddingModel = "openai:text-embedding-3-small"
EMBEDDING_MODEL_TO_DIMENSIONS: dict[SupportedEmbeddingModel, int] = {
    "mixedbread:mxbai-embed-large-v1": 1024,
    "mixedbread:mxbai-embed-2d-large-v1": 768,
    "mixedbread:deepset-mxbai-embed-de-large-v1": 1024,
    "mixedbread:mxbai-embed-xsmall-v1": 384,
    "openai:text-embedding-3-large": 3072,
    "openai:text-embedding-3-small": 1536,
    "openai:text-embedding-ada-002": 1536,
    "togetherai:BAAI/bge-base-en-v1.5": 768,
    "togetherai:BAAI/bge-large-en-v1.5": 1024,
    "togetherai:BAAI/bge-m3": 1024,
    "togetherai:WhereIsAI/UAE-Large-V1": 1024,
    "togetherai:togethercomputer/m2-bert-80M-8k-retrieval": 768,
    "togetherai:Alibaba-NLP/gte-Qwen2-7B-instruct": 3584,
    "togetherai:Alibaba-NLP/gte-modernbert-base": 768,
    "deepinfra:Qwen/Qwen3-Embedding-8B": 4096,
    "deepinfra:Qwen/Qwen3-Embedding-4B": 2560,
    "deepinfra:Qwen/Qwen3-Embedding-0.6B": 1024,
    "deepinfra:BAAI/bge-m3": 1024,
    "deepinfra:BAAI/bge-base-en-v1.5": 768,
    "deepinfra:BAAI/bge-large-en-v1.5": 1024,
    "deepinfra:BAAI/bge-en-icl": 4096,
    "deepinfra:intfloat/e5-base-v2": 768,
    "deepinfra:intfloat/e5-large-v2": 1024,
    "deepinfra:intfloat/multilingual-e5-large": 1024,
    "deepinfra:sentence-transformers/all-MiniLM-L6-v2": 384,
    "deepinfra:sentence-transformers/all-MiniLM-L12-v2": 384,
    "deepinfra:sentence-transformers/all-mpnet-base-v2": 768,
    "deepinfra:thenlper/gte-base": 768,
    "deepinfra:thenlper/gte-large": 1024,
    "deepinfra:google/embeddinggemma-300m": 768,
    "gemini:gemini-embedding-001": 3072,
    "cloudflare:@cf/baai/bge-small-en-v1.5": 384,
    "cloudflare:@cf/baai/bge-base-en-v1.5": 768,
    "cloudflare:@cf/baai/bge-large-en-v1.5": 1024,
    "cloudflare:@cf/baai/bge-m3": 1024,
    "cloudflare:@cf/qwen/qwen3-embedding-0.6b": 1024,
    "cloudflare:@cf/google/embeddinggemma-300m": 768,
    "cloudflare:@cf/pfnet/plamo-embedding-1b": 1024,
    "cohere:embed-v4.0": 1536,
    "cohere:embed-english-v3.0": 1024,
    "cohere:embed-english-light-v3.0": 384,
    "cohere:embed-multilingual-v3.0": 1024,
    "cohere:embed-multilingual-light-v3.0": 384,
    "jinaai:jina-embeddings-v4": 2048,
    "jinaai:jina-embeddings-v3": 1024,
    "jinaai:jina-code-embeddings-1.5b": 1536,
    "jinaai:jina-code-embeddings-0.5b": 896,
    "jinaai:jina-embeddings-v2-base-en": 768,
    "jinaai:jina-embeddings-v2-base-code": 768,
    "voyageai:voyage-4-large": 1024,
    "voyageai:voyage-4": 1024,
    "voyageai:voyage-4-lite": 1024,
    "voyageai:voyage-4-nano": 1024,
    "voyageai:voyage-3-large": 1024,
    "voyageai:voyage-3.5": 1024,
    "voyageai:voyage-3.5-lite": 1024,
    "voyageai:voyage-3": 1024,
    "voyageai:voyage-3-lite": 1024,
    "voyageai:voyage-code-3": 1024,
    "voyageai:voyage-finance-2": 1024,
    "voyageai:voyage-law-2": 1024,
    "voyageai:voyage-code-2": 1536,
    "voyageai:voyage-multilingual-2": 1024,
    "voyageai:voyage-multimodal-3": 1024,
    "mistral:codestral-embed-2505": 1536,
    "mistral:mistral-embed": 1024,
    "nomic:nomic-embed-text-v1.5": 768,
    "nomic:nomic-embed-text-v1": 768,
}

PROVIDER_API_KEYS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "voyageai": ["VOYAGE_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "jina": ["JINA_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "mixedbread": ["MIXEDBREAD_API_KEY"],
    "nomic": ["NOMIC_API_KEY"],
    "deepinfra": ["DEEPINFRA_API_KEY"],
    "cloudflare": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
}


def check_api_key(provider: str) -> None:
    """Check that all required API keys are set for the given provider.

    For providers with multiple key options (e.g. Gemini), at least one must be set.
    For providers that require multiple keys (e.g. Cloudflare), all must be set.

    Raises:
        ValueError: If the provider is unknown or required keys are missing.
    """
    provider = provider.lower()
    if provider not in PROVIDER_API_KEYS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available providers: {', '.join(PROVIDER_API_KEYS)}"
        )

    required_keys = PROVIDER_API_KEYS[provider]

    # Cloudflare requires ALL keys; others need at least one
    if provider == "cloudflare":
        missing = [k for k in required_keys if not os.environ.get(k)]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s) for '{provider}': "
                + ", ".join(missing)
            )
    else:
        if not any(os.environ.get(k) for k in required_keys):
            keys_str = (
                " or ".join(required_keys)
                if len(required_keys) > 1
                else required_keys[0]
            )
            raise ValueError(
                f"No API key found for '{provider}'. Please set {keys_str}."
            )


class EmbeddingsService:
    def __init__(
        self,
        model_to_dims: dict[SupportedEmbeddingModel, int] | None = None,
    ) -> None:
        self._model_to_dims = model_to_dims or EMBEDDING_MODEL_TO_DIMENSIONS
        self._client: Client = Client()

    def resolve_dims(self, model: SupportedEmbeddingModel | None = None) -> int:
        mod: SupportedEmbeddingModel = model or DEFAULT_EMBEDDING_MODEL
        return self._model_to_dims[mod]

    @overload
    async def embed(
        self, input: str, model: SupportedEmbeddingModel | None = None
    ) -> list[float]: ...

    @overload
    async def embed(
        self, input: list[str], model: SupportedEmbeddingModel | None = None
    ) -> list[list[float]]: ...

    async def embed(
        self, input: str | list[str], model: SupportedEmbeddingModel | None = None
    ) -> list[float] | list[list[float]]:
        mod: SupportedEmbeddingModel = model or DEFAULT_EMBEDDING_MODEL
        provider, _ = mod.split(":", 1)
        check_api_key(provider)
        response = await self._client.aembed(input)
        if isinstance(input, str):
            return response.embeddings[0]
        else:
            return response.embeddings


embeddings_service = EmbeddingsService()


@functools.lru_cache
def get_embeddings_service() -> EmbeddingsService:
    return embeddings_service
