"""Shared pytest fixtures and environment setup.

Pre-populates env vars that several pydantic-settings classes mark as
required so that simply importing the modules in tests doesn't blow up
when running outside of a configured deployment.
"""

from __future__ import annotations

import os

# Defaults are applied at import time so any module that calls a cached
# settings factory at top-level still works. Tests that need different
# values should call ``settings.cache_clear()`` and patch os.environ.
_DEFAULT_ENV = {
    "DEFAULT_INDEX_MONGO_DB": "test_db",
    "DEFAULT_INDEX_MONGO_URI": "mongodb://user:pass@localhost:27017",
    "DEFAULT_INDEX_MONGO_COLLECTION": "test_coll",
    "DEFAULT_INDEX_QDRANT_URL": "http://localhost:6333",
    "DEFAULT_INDEX_PINECONE_API_KEY": "test-pinecone-key",
}
for k, v in _DEFAULT_ENV.items():
    os.environ.setdefault(k, v)
