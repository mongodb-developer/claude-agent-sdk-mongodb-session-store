"""Constructor and validation tests. These need no MongoDB server."""

from __future__ import annotations

from typing import Any

import pytest
from claude_agent_sdk._internal.session_store_validation import _store_implements
from pymongo import AsyncMongoClient

from claude_agent_sdk_mongodb_session_store import (
    MongoDBSessionStore,
    MongoDBSessionStoreOptions,
)

Client = AsyncMongoClient[dict[str, Any]]


class TestConstructor:
    @pytest.mark.anyio
    async def test_requires_client(self) -> None:
        with pytest.raises(ValueError, match="requires 'client'"):
            MongoDBSessionStore()

    @pytest.mark.anyio
    async def test_store_implements_required_methods(
        self, offline_client: Client
    ) -> None:
        """SessionStore is not @runtime_checkable; probe via _store_implements()."""
        store = MongoDBSessionStore(client=offline_client, db_name="db")
        assert _store_implements(store, "append")
        assert _store_implements(store, "load")

    @pytest.mark.anyio
    async def test_default_database_from_uri(self, offline_client: Client) -> None:
        store = MongoDBSessionStore(client=offline_client)
        assert store._db.name == "offline"

    @pytest.mark.anyio
    async def test_positional_args(self, offline_client: Client) -> None:
        store = MongoDBSessionStore(offline_client, "db", "ents", "sums")
        assert store._db.name == "db"
        assert store._entries.name == "ents"
        assert store._summaries.name == "sums"

    @pytest.mark.anyio
    async def test_options_take_precedence(self, offline_client: Client) -> None:
        other: Client = AsyncMongoClient("mongodb://localhost:1/other", connect=False)
        try:
            store = MongoDBSessionStore(
                other,
                "ignored_db",
                "ignored_entries",
                options=MongoDBSessionStoreOptions(
                    client=offline_client,
                    db_name="db",
                    entries_collection="ents",
                    summaries_collection="sums",
                ),
            )
        finally:
            await other.close()
        assert store._db.client is offline_client
        assert store._db.name == "db"
        assert store._entries.name == "ents"
        assert store._summaries.name == "sums"


class TestCollectionNames:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "name",
        ["", "a..b", "x.", ".x", "bad$col", "nul\x00", "system.foo", "system.users"],
    )
    @pytest.mark.parametrize("field", ["entries_collection", "summaries_collection"])
    async def test_rejects_invalid_collection_name(
        self, offline_client: Client, field: str, name: str
    ) -> None:
        # Always ValueError (never pymongo's InvalidName), naming the field.
        with pytest.raises(ValueError, match=field):
            MongoDBSessionStore(client=offline_client, db_name="db", **{field: name})

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "name", ["ok", "ns.sub", "with space", "semi;colon", "Ünï"]
    )
    async def test_accepts_names_mongodb_allows(
        self, offline_client: Client, name: str
    ) -> None:
        store = MongoDBSessionStore(
            client=offline_client,
            db_name="db",
            entries_collection=name,
            summaries_collection=f"{name}_s",
        )
        assert store._entries.name == name
        assert store._summaries.name == f"{name}_s"
