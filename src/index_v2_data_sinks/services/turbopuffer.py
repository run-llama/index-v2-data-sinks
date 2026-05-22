"""Turbopuffer export target: connection resolution and namespace lifecycle.

Wraps :class:`TurbopufferIndexV2Settings` so callers don't need to know about
environment variable layout or per-target overrides. Each target id resolves
to a tpuf namespace; namespaces are cheap and the platform fans every export
config out to its own namespace, which is the recommended multi-tenant
pattern for Turbopuffer.
"""

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError
from turbopuffer import (
    RateLimitError as _TpufRateLimitError,  # ty: ignore[unresolved-import]
)

from ..configs.turbopuffer import (
    TurbopufferIndexV2Settings,
    get_turbopuffer_indexv2_settings,
)

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from turbopuffer import AsyncTurbopuffer  # ty: ignore[unresolved-import]
    from turbopuffer.lib.namespace import (
        AsyncNamespace,  # ty: ignore[unresolved-import]
    )
    from turbopuffer.types.custom import Filter  # ty: ignore[unresolved-import]


# Tpuf attribute names for the two fields that scope every retrieval filter
# to a single tenant + export config. Defined here so the export-side write
# and retrieval-side filter construction reference the same string — a
# typo on either side would otherwise produce silently-empty results.
TENANT_ID_FIELD = "tenant_id"
EXPORT_CONFIG_ID_FIELD = "export_config_id"
# The export side pre-computes the metadata-fields hash and stores it here
# so the snapshot diff doesn't have to re-hash a JSON-round-tripped dict.
# Avoids hash-stability drift if the metadata schema ever widens past
# JSON-stable primitive types.
METADATA_FIELDS_HASH_FIELD = "metadata_fields_hash"

# Prefix for the flat tpuf attributes that mirror keys of ``metadata_fields``.
# Tpuf attribute types are flat (no object / nested type), so to support
# Mongo-parity filtering on metadata we promote each metadata key to its own
# top-level attribute. The prefix avoids colliding with the structured
# top-level attributes (``tenant_id``, ``export_config_id``, etc.) and the
# JSON-string round-trip stored under ``metadata_fields``. Dots in metadata
# keys (e.g. ``extract.page_1.foo`` from extract flattening) become ``__``
# because tpuf attribute names cannot contain dots.
METADATA_ATTR_PREFIX = "mf__"


def metadata_attr_name(key: str) -> str:
    """Return the tpuf attribute name that holds ``metadata_fields[key]``.

    Used by both the export target (when emitting flattened attributes on
    write) and the retrieval service (when translating custom filters).
    """
    return f"{METADATA_ATTR_PREFIX}{key.replace('.', '__')}"


# Typed Filter constructors. Tpuf's ``Filter`` is a Union of typed tuples
# keyed on Literal operator strings (``Tuple[str, Literal["Eq"], Any]``,
# etc.), and mypy / ty don't narrow inline tuple literals into a specific
# union arm because the operator string is inferred as plain ``str``. These
# helpers solve that without per-call ``cast``: the tight return annotation
# lets the type checker narrow the inline literal at the helper's return
# site, and the call site reads as a small DSL.


def filter_eq(field: str, value: Any) -> "tuple[str, Literal['Eq'], Any]":
    return (field, "Eq", value)


def filter_noteq(field: str, value: Any) -> "tuple[str, Literal['NotEq'], Any]":
    return (field, "NotEq", value)


def filter_in(
    field: str, values: Sequence[Any]
) -> "tuple[str, Literal['In'], Sequence[Any]]":
    return (field, "In", values)


def filter_gt(field: str, value: Any) -> "tuple[str, Literal['Gt'], Any]":
    return (field, "Gt", value)


def filter_and(
    clauses: "Sequence[Filter]",
) -> "tuple[Literal['And'], Sequence[Filter]]":
    return ("And", clauses)


TURBOPUFFER_INDEX_V2_NOT_CONFIGURED_MESSAGE = (
    "Turbopuffer index V2 is not configured for this environment. "
    "Set DEFAULT_INDEX_TPUF_API_KEY (and optionally DEFAULT_INDEX_TPUF_REGION, "
    "DEFAULT_INDEX_TPUF_NAMESPACE_PREFIX)."
)


class TurbopufferIndexV2NotConfiguredError(RuntimeError):
    """Raised when Turbopuffer index V2 settings are missing required values.

    HTTP boundary callers should translate this into a 503 so the response
    is actionable rather than a generic 500 with a pydantic trace.
    """


class TurbopufferRateLimitError(RuntimeError):
    """Raised when Turbopuffer rejects a request with HTTP 429.

    Wraps the SDK's ``RateLimitError`` so the HTTP boundary can map it to
    an actionable 429 response and so Temporal activities on the export
    side surface it as a recognizable transient failure rather than an
    opaque ``APIStatusError``. The underlying SDK exception is chained
    via ``raise from``.
    """


class SnapshotEnumerationCapExceeded(RuntimeError):
    """Raised when ``list_snapshots`` exhausts its pagination ceiling.

    Hitting the cap means the snapshot is incomplete — the diff service
    would interpret missing files as needing re-export and would repeat
    that work every cycle. Raising forces the caller to handle the
    over-cap config explicitly (alert, manual reindex, or move that
    config off the tpuf path) rather than silently corrupting the diff.
    """


@dataclass(frozen=True)
class TurbopufferConnectionTarget:
    """Resolved Turbopuffer connection coordinates for a single export target."""

    api_key: str
    region: str
    namespace: str


# Bounded cache because the keyspace can grow with the multi-tenant
# wiring: ``region_map`` / ``namespace_map`` and any future BYO-credentials
# path (each export config bringing its own api_key) push distinct
# ``(api_key, region)`` pairs through this cache. Today's platform-only
# deployment uses ~one entry; 128 leaves significant headroom for BYO
# without letting the cache grow unboundedly across long-running services.
_CLIENT_CACHE_MAXSIZE = 128


@lru_cache(maxsize=_CLIENT_CACHE_MAXSIZE)
def _get_turbopuffer_async_client(api_key: str, region: str) -> "AsyncTurbopuffer":
    """Return a cached AsyncTurbopuffer client keyed by (api_key, region).

    The export and retrieval paths are both async; using the SDK's native
    async client lets us ``await`` network calls directly instead of
    dispatching them onto a thread pool, which would block the event loop
    under concurrent ingest.
    """
    import turbopuffer  # ty: ignore[unresolved-import]

    return turbopuffer.AsyncTurbopuffer(api_key=api_key, region=region)


class TurbopufferService:
    """Resolves a target id to concrete Turbopuffer connection parameters.

    Accepts an optional :class:`TurbopufferIndexV2Settings` for testability;
    when omitted the cached singleton from
    :func:`get_turbopuffer_indexv2_settings` is used.
    """

    def __init__(self, settings: TurbopufferIndexV2Settings | None = None) -> None:
        self._settings = settings

    @property
    def settings(self) -> TurbopufferIndexV2Settings:
        if self._settings is None:
            try:
                self._settings = get_turbopuffer_indexv2_settings()
            except ValidationError as e:
                raise TurbopufferIndexV2NotConfiguredError(
                    TURBOPUFFER_INDEX_V2_NOT_CONFIGURED_MESSAGE
                ) from e
        return self._settings

    def resolve(self, target_id: str) -> TurbopufferConnectionTarget:
        """Return connection coordinates for ``target_id``."""
        return TurbopufferConnectionTarget(
            api_key=self.settings.api_key.get_secret_value(),
            region=self.settings.resolve_region(target_id),
            namespace=self.settings.resolve_namespace(target_id),
        )

    def get_client(self, target_id: str) -> "AsyncTurbopuffer":
        """Return a cached AsyncTurbopuffer client for ``target_id``."""
        target = self.resolve(target_id)
        return _get_turbopuffer_async_client(target.api_key, target.region)

    def get_namespace(self, target_id: str) -> "AsyncNamespace":
        """Return a tpuf AsyncNamespace handle for ``target_id``.

        Namespaces are created lazily on first write by Turbopuffer itself,
        so there's no explicit ``ensure`` step.
        """
        target = self.resolve(target_id)
        return self.get_client(target_id).namespace(target.namespace)


tpuf_service = TurbopufferService()


def get_turbopuffer_service() -> TurbopufferService:
    """Return the module-level singleton."""
    return tpuf_service


@asynccontextmanager
async def translate_tpuf_errors() -> AsyncGenerator[None]:
    """Translate tpuf SDK errors into our typed errors at the call boundary.

    Currently maps ``RateLimitError`` (HTTP 429) to
    :class:`TurbopufferRateLimitError`. Other ``APIStatusError`` subclasses
    propagate untouched — Temporal activity retry policies and the HTTP
    error mapper handle them generically.

    Use as a wrapper around tpuf SDK calls in code paths that want the
    typed 429 (the retrieval endpoint maps it to HTTP 429; Temporal
    activities can recognize it as a transient and back off cleanly).
    """
    try:
        yield
    except _TpufRateLimitError as e:
        raise TurbopufferRateLimitError(str(e)) from e
