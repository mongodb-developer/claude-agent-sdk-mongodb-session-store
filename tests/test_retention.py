"""Live tests for ``delete_inactive()``: whole sessions expire, never parts."""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import pytest
from claude_agent_sdk import SessionKey

from claude_agent_sdk_mongodb_session_store import MongoDBSessionStore

from .conftest import StoreFactory

DAY_MS = 86_400_000


async def _age(store: MongoDBSessionStore, days: int, **match: Any) -> None:
    """Backdate the matching entries and counters by ``days``, as if nothing
    had been appended to them since."""
    aged = {"$set": {"mtime": int(time.time() * 1000) - days * DAY_MS}}
    await store._entries.update_many(match, aged)
    await store._counters.update_many({f"_id.{k}": v for k, v in match.items()}, aged)


def _key(session_id: str, project_key: str = "proj", **extra: str) -> SessionKey:
    return {"project_key": project_key, "session_id": session_id, **extra}  # type: ignore[typeddict-item]


class TestDeleteInactive:
    @pytest.mark.anyio
    async def test_deletes_idle_sessions_whole_and_keeps_active_ones_whole(
        self, make_store: StoreFactory
    ) -> None:
        store = await make_store()
        sub = "subagents/agent-1"

        # Idle: every entry, main and subagent, is 40 days old.
        await store.append(_key("idle"), [{"type": "user", "customTitle": "idle"}])
        await store.append(_key("idle", subpath=sub), [{"type": "user"}])
        await _age(store, 40, session_id="idle")

        # Active: 40 days of history plus one entry today. Must survive intact,
        # not lose its old entries.
        await store.append(_key("active"), [{"type": "user", "n": 1}])
        await _age(store, 40, session_id="active")
        await store.append(_key("active"), [{"type": "user", "n": 2}])

        # Main transcript idle, but a subagent wrote today: still active.
        await store.append(_key("sub_active"), [{"type": "user"}])
        await _age(store, 40, session_id="sub_active")
        await store.append(_key("sub_active", subpath=sub), [{"type": "user"}])

        # Only subagent entries, all idle: nothing else will ever clean these.
        await store.append(_key("orphan", subpath=sub), [{"type": "user"}])
        await _age(store, 40, session_id="orphan")

        # Idle session in another project: the sweep covers every project.
        await store.append(_key("idle", project_key="other"), [{"type": "user"}])
        await _age(store, 40, project_key="other")

        deleted = await store.delete_inactive(timedelta(days=30))

        assert deleted == 3
        assert await store.load(_key("idle")) is None
        assert await store.load(_key("idle", subpath=sub)) is None
        assert await store.load(_key("orphan", subpath=sub)) is None
        assert await store.load(_key("idle", project_key="other")) is None
        assert await store.load(_key("active")) == [
            {"type": "user", "n": 1},
            {"type": "user", "n": 2},
        ]
        assert await store.load(_key("sub_active")) == [{"type": "user"}]
        assert await store.load(_key("sub_active", subpath=sub)) == [{"type": "user"}]

        summaries = await store.list_session_summaries("proj")
        assert sorted(s["session_id"] for s in summaries) == ["active", "sub_active"]
        counters = sorted(
            [
                (d["_id"]["session_id"], d["_id"]["subpath"])
                async for d in store._counters.find({})
            ]
        )
        assert counters == [("active", ""), ("sub_active", ""), ("sub_active", sub)]

    @pytest.mark.anyio
    async def test_nothing_idle_deletes_nothing(self, make_store: StoreFactory) -> None:
        store = await make_store()
        await store.append(_key("s"), [{"type": "user"}])
        assert await store.delete_inactive(timedelta(days=30)) == 0
        assert await store.load(_key("s")) == [{"type": "user"}]

    @pytest.mark.anyio
    async def test_removes_counters_left_by_delete_without_counting_them(
        self, make_store: StoreFactory
    ) -> None:
        store = await make_store()
        await store.append(_key("gone"), [{"type": "user"}])
        await store.delete(_key("gone"))
        await _age(store, 40, session_id="gone")

        assert await store.delete_inactive(timedelta(days=30)) == 0
        assert await store._counters.count_documents({}) == 0

    @pytest.mark.anyio
    async def test_keeps_the_counter_of_a_transcript_appended_to_during_the_sweep(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep found the session idle, then an append reserved positions
        before the sweep deleted it. The counter must survive so the append's
        entries, if they land, keep their place."""
        store = await make_store()
        await store.append(_key("s"), [{"type": "user"}])
        await _age(store, 40, session_id="s")
        real_delete_many = store._entries.delete_many

        async def delete_many(*args: Any, **kwargs: Any) -> Any:
            await store._reserve_positions(_key("s"), "", 1, int(time.time() * 1000))
            return await real_delete_many(*args, **kwargs)

        monkeypatch.setattr(store._entries, "delete_many", delete_many)
        await store.delete_inactive(timedelta(days=30))

        [counter] = [d async for d in store._counters.find({})]
        assert counter["last_position"] == 2
