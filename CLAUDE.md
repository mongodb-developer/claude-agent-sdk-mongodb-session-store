# Workflow

```bash
# Install / sync the dev environment
uv sync

# Lint and style (check and fix automatically)
uv run ruff check claude_agent_sdk_mongodb_session_store/ tests/ --fix
uv run ruff format claude_agent_sdk_mongodb_session_store/ tests/

# Typecheck
uv run mypy claude_agent_sdk_mongodb_session_store/ tests/

# All pre-commit hooks (ruff, mypy, uv-lock, whitespace)
uv run pre-commit run --all-files

# Run all tests (live tests need a MongoDB server)
MONGODB_URI=mongodb://localhost:27017 uv run pytest

# Run specific test file
uv run pytest tests/test_mongodb_session_store.py
```

# Codebase Structure

- `claude_agent_sdk_mongodb_session_store/` - Main package
  - `__init__.py` - Public exports (`MongoDBSessionStore`, `MongoDBSessionStoreOptions`)
  - `_store.py` - The `SessionStore` implementation
- `tests/` - Live-MongoDB tests, gated on `MONGODB_URI`
