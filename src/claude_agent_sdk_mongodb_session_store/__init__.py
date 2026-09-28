"""MongoDB-backed :class:`~claude_agent_sdk.SessionStore` for the Claude Agent SDK."""

from importlib.metadata import PackageNotFoundError, version

from ._store import MongoDBSessionStore, MongoDBSessionStoreOptions

try:
    __version__ = version("claude-agent-sdk-mongodb-session-store")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = ["MongoDBSessionStore", "MongoDBSessionStoreOptions", "__version__"]
