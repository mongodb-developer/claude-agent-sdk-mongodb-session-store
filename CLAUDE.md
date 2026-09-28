# Workflow

```bash
# Install / sync the dev environment
uv sync

# Lint and style (check and fix automatically)
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/

# Typecheck
uv run mypy src/

# All pre-commit hooks (ruff, mypy, uv-lock, whitespace)
uv run pre-commit run --all-files

# Run all tests (live tests need a MongoDB server)
SESSION_STORE_MONGODB_URL=mongodb://localhost:27017 uv run pytest

# Run specific test file
uv run pytest tests/test_mongodb_session_store.py
```

# Codebase Structure

- `src/claude_agent_sdk_mongodb_session_store/` - Main package
  - `__init__.py` - Public exports (`MongoDBSessionStore`, `MongoDBSessionStoreOptions`)
  - `_store.py` - The `SessionStore` implementation
- `tests/` - Live-MongoDB tests, gated on `SESSION_STORE_MONGODB_URL`
