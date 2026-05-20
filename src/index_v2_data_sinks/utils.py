import hashlib
import json

from pydantic import BaseModel


def compute_metadata_hash(metadata: BaseModel | dict | None) -> str | None:
    """Compute a deterministic SHA-256 hash of a metadata object.

    Accepts a Pydantic BaseModel, a raw dict, or None.
    Returns None when metadata is None or empty, so that (None vs None)
    compares as unchanged.
    """
    if metadata is None:
        return None

    if isinstance(metadata, BaseModel):
        serializable = metadata.model_dump()
    else:
        serializable = metadata

    if not serializable:
        return None

    canonical = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
