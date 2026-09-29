"""Shared fixtures.

Tests that need a server request the ``client`` (or ``mongodb_url``) fixture,
which skips unless ``SESSION_STORE_MONGODB_URL`` is set. Tests that only
construct a store use ``offline_client``, which never connects, so they run
everywhere.
"""

from __future__ import annotations

import itertools
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from pymongo import AsyncMongoClient

from claude_agent_sdk_mongodb_session_store import MongoDBSessionStore

MONGODB_URL_ENV = "SESSION_STORE_MONGODB_URL"


@pytest.fixture
def anyio_backend() -> str:
    # ``pymongo``'s async API has no trio backend.
    return "asyncio"


@pytest.fixture(scope="session")
def mongodb_url() -> str:
    url = os.environ.get(MONGODB_URL_ENV)
    if not url:
        pytest.skip(
            f"live MongoDB e2e: set {MONGODB_URL_ENV} (e.g. mongodb://localhost:27017)"
        )
    return url


@pytest.fixture
async def client(mongodb_url: str) -> AsyncIterator[AsyncMongoClient[dict[str, Any]]]:
    c: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(mongodb_url)
    try:
        yield c
    finally:
        await c.close()


@pytest.fixture
async def db_name(client: AsyncMongoClient[dict[str, Any]]) -> AsyncIterator[str]:
    name = f"claude_test_{uuid.uuid4().hex[:8]}"
    try:
        yield name
    finally:
        await client.drop_database(name)


@pytest.fixture
async def offline_client() -> AsyncIterator[AsyncMongoClient[dict[str, Any]]]:
    """A client that is never used for I/O.

    ``AsyncMongoClient`` does not connect on construction, so constructor and
    validation tests can run without a server.
    """
    c: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
        "mongodb://localhost:1/offline", connect=False
    )
    try:
        yield c
    finally:
        await c.close()


StoreFactory = Callable[..., Awaitable[MongoDBSessionStore]]


@pytest.fixture
def make_store(client: AsyncMongoClient[dict[str, Any]], db_name: str) -> StoreFactory:
    """Build schema-initialized stores in the per-test database.

    ``make_store()`` gives a store on its own fresh collections. Pass
    ``prefix=`` to get a second instance on the *same* collections as an
    earlier call, which simulates another process sharing the database.
    """
    counter = itertools.count()

    async def factory(prefix: str | None = None) -> MongoDBSessionStore:
        p = prefix if prefix is not None else f"c{next(counter)}"
        store = MongoDBSessionStore(
            client=client,
            db_name=db_name,
            entries_collection=f"{p}_entries",
            summaries_collection=f"{p}_summaries",
            counters_collection=f"{p}_counters",
        )
        await store.create_schema()
        return store

    return factory
