"""Constructor and validation tests. These need no MongoDB server."""

from __future__ import annotations

from datetime import timedelta
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
        store = MongoDBSessionStore(offline_client, "db", "ents", "sums", "ctrs")
        assert store._db.name == "db"
        assert store._entries.name == "ents"
        assert store._summaries.name == "sums"
        assert store._counters.name == "ctrs"

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
                    counters_collection="ctrs",
                ),
            )
        finally:
            await other.close()
        assert store._db.client is offline_client
        assert store._db.name == "db"
        assert store._entries.name == "ents"
        assert store._summaries.name == "sums"
        assert store._counters.name == "ctrs"


class TestCollectionNames:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "name",
        ["", "a..b", "x.", ".x", "bad$col", "nul\x00", "system.foo", "system.users"],
    )
    @pytest.mark.parametrize(
        "field", ["entries_collection", "summaries_collection", "counters_collection"]
    )
    async def test_rejects_invalid_collection_name(
        self, offline_client: Client, field: str, name: str
    ) -> None:
        # Always ValueError (never pymongo's InvalidName), naming the field.
        names: dict[str, Any] = {field: name}
        with pytest.raises(ValueError, match=field):
            MongoDBSessionStore(client=offline_client, db_name="db", **names)

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


class TestDeleteInactiveValidation:
    @pytest.mark.anyio
    @pytest.mark.parametrize("older_than", [timedelta(0), timedelta(days=-1)])
    async def test_rejects_non_positive_age(
        self, offline_client: Client, older_than: timedelta
    ) -> None:
        # Raises before any I/O: the offline client would fail to connect.
        store = MongoDBSessionStore(client=offline_client, db_name="db")
        with pytest.raises(ValueError, match="older_than"):
            await store.delete_inactive(older_than)


class TestKeyValidation:
    """Key fields go into query filters verbatim, so a non-string (e.g. a
    dict such as ``{"$ne": ""}``) would act as a query operator and read or
    delete across sessions and tenants. Every method must reject it before
    any I/O: the offline client would fail to connect otherwise."""

    BAD_VALUES: list[Any] = [{"$ne": ""}, {"$regex": ".*"}, ["a"], 1, None]

    @pytest.mark.anyio
    @pytest.mark.parametrize("bad", BAD_VALUES)
    @pytest.mark.parametrize("field", ["project_key", "session_id"])
    @pytest.mark.parametrize("method", ["append", "load", "delete", "list_subkeys"])
    async def test_key_methods_reject_non_string(
        self, offline_client: Client, method: str, field: str, bad: Any
    ) -> None:
        store = MongoDBSessionStore(client=offline_client, db_name="db")
        key: dict[str, Any] = {"project_key": "p", "session_id": "s", field: bad}
        args: tuple[Any, ...] = (
            (key, [{"type": "user"}]) if method == "append" else (key,)
        )
        with pytest.raises(TypeError, match=field):
            await getattr(store, method)(*args)

    @pytest.mark.anyio
    @pytest.mark.parametrize("bad", [{"$ne": ""}, ["a"], 1])
    @pytest.mark.parametrize("method", ["append", "load", "delete"])
    async def test_key_methods_reject_non_string_subpath(
        self, offline_client: Client, method: str, bad: Any
    ) -> None:
        store = MongoDBSessionStore(client=offline_client, db_name="db")
        key: dict[str, Any] = {"project_key": "p", "session_id": "s", "subpath": bad}
        args: tuple[Any, ...] = (
            (key, [{"type": "user"}]) if method == "append" else (key,)
        )
        with pytest.raises(TypeError, match="subpath"):
            await getattr(store, method)(*args)

    @pytest.mark.anyio
    @pytest.mark.parametrize("field", ["project_key", "session_id"])
    @pytest.mark.parametrize("method", ["append", "load", "delete", "list_subkeys"])
    async def test_key_methods_reject_empty_string(
        self, offline_client: Client, method: str, field: str
    ) -> None:
        store = MongoDBSessionStore(client=offline_client, db_name="db")
        key: dict[str, Any] = {"project_key": "p", "session_id": "s", field: ""}
        args: tuple[Any, ...] = (
            (key, [{"type": "user"}]) if method == "append" else (key,)
        )
        with pytest.raises(ValueError, match=field):
            await getattr(store, method)(*args)

    @pytest.mark.anyio
    @pytest.mark.parametrize("bad", BAD_VALUES)
    @pytest.mark.parametrize("method", ["list_sessions", "list_session_summaries"])
    async def test_project_methods_reject_non_string(
        self, offline_client: Client, method: str, bad: Any
    ) -> None:
        store = MongoDBSessionStore(client=offline_client, db_name="db")
        with pytest.raises(TypeError, match="project_key"):
            await getattr(store, method)(bad)

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["list_sessions", "list_session_summaries"])
    async def test_project_methods_reject_empty_string(
        self, offline_client: Client, method: str
    ) -> None:
        store = MongoDBSessionStore(client=offline_client, db_name="db")
        with pytest.raises(ValueError, match="project_key"):
            await getattr(store, method)("")

    @pytest.mark.anyio
    async def test_append_with_empty_batch_still_validates(
        self, offline_client: Client
    ) -> None:
        store = MongoDBSessionStore(client=offline_client, db_name="db")
        with pytest.raises(TypeError, match="session_id"):
            await store.append({"project_key": "p", "session_id": {"$ne": ""}}, [])  # type: ignore[typeddict-item]
