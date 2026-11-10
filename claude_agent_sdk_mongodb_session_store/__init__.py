"""MongoDB-backed :class:`~claude_agent_sdk.SessionStore` for the Claude Agent SDK."""

from ._store import (
    EntryTooLargeError,
    MongoDBSessionStore,
    MongoDBSessionStoreOptions,
)
from ._version import __version__

__all__ = [
    "EntryTooLargeError",
    "MongoDBSessionStore",
    "MongoDBSessionStoreOptions",
    "__version__",
]
