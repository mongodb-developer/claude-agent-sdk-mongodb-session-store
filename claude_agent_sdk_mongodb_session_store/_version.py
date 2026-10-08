"""Package version, shared by ``__init__`` and the driver handshake metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("claude-agent-sdk-mongodb-session-store")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"
