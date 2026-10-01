"""Live tests pinning the query plans the adapter relies on for performance.

Each test runs the real store method with the database profiler on, then reads
the server's own record of how the command executed.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import pytest
from claude_agent_sdk import SessionKey

from claude_agent_sdk_mongodb_session_store import MongoDBSessionStore

from .conftest import StoreFactory, entry


@asynccontextmanager
async def _profiling(store: MongoDBSessionStore) -> AsyncIterator[None]:
    """Record every operation on the store's database while in the block."""
    await store._db.command("profile", 2)
    try:
        yield
    finally:
        await store._db.command("profile", 0)


async def _profiled(
    store: MongoDBSessionStore, collection: Any, **match: Any
) -> list[dict[str, Any]]:
    """Profiler records for operations on ``collection``, oldest first."""
    ns = f"{store._db.name}.{collection.name}"
    cursor = store._db["system.profile"].find({"ns": ns, **match}).sort("ts", 1)
    return [dict(r) async for r in cursor]


class TestListSessionsPlan:
    @pytest.mark.anyio
    async def test_list_sessions_reads_one_index_key_per_session(
        self, make_store: StoreFactory
    ) -> None:
        """``list_sessions()`` must not scan every entry in the project.

        With a ``$sort`` + ``$group``/``$first`` over a matching index, MongoDB
        jumps from one session's newest key to the next session's
        (``DISTINCT_SCAN``) and never fetches a document.
        """
        n_sessions, n_entries = 5, 200
        store = await make_store()
        for s in range(n_sessions):
            key: SessionKey = {"project_key": "proj", "session_id": f"s{s}"}
            await store.append(
                key, [entry({"type": "user", "n": i}) for i in range(n_entries)]
            )
            # Subagent entries must neither be listed nor examined.
            await store.append(
                {**key, "subpath": "subagents/a"}, [{"type": "user"}] * 50
            )

        async with _profiling(store):
            listed = await store.list_sessions("proj")

        assert sorted(e["session_id"] for e in listed) == [
            f"s{s}" for s in range(n_sessions)
        ]
        [record] = await _profiled(
            store, store._entries, **{"command.aggregate": {"$exists": True}}
        )
        assert record["planSummary"].startswith("DISTINCT_SCAN"), record["planSummary"]
        assert record["docsExamined"] == 0
        # A few keys per session at most, never one per entry.
        assert record["keysExamined"] <= 2 * n_sessions + 1, record["keysExamined"]


class TestSummariesPlan:
    @pytest.mark.anyio
    async def test_summary_rewrite_touches_no_index_keys(
        self, make_store: StoreFactory
    ) -> None:
        """Every main-transcript append rewrites the session's summary with a
        new ``mtime``. No index may include a field that changes on rewrite,
        or each append pays for an index key delete + insert."""
        store = await make_store()
        key: SessionKey = {"project_key": "proj", "session_id": "s"}
        await store.append(key, [entry({"type": "user", "customTitle": "t"})])

        async with _profiling(store):
            for i in range(5):
                await store.append(key, [entry({"type": "user", "n": i})])

        writes = await _profiled(store, store._summaries, op="update")
        assert len(writes) == 5
        assert [(w["keysInserted"], w["keysDeleted"]) for w in writes] == [(0, 0)] * 5

    @pytest.mark.anyio
    async def test_list_session_summaries_reads_only_that_project(
        self, make_store: StoreFactory
    ) -> None:
        store = await make_store()
        for project, n_sessions in [("proj", 3), ("other", 20)]:
            for s in range(n_sessions):
                await store.append(
                    {"project_key": project, "session_id": f"s{s}"},
                    [{"type": "user"}],
                )

        async with _profiling(store):
            summaries = await store.list_session_summaries("proj")

        assert sorted(s["session_id"] for s in summaries) == ["s0", "s1", "s2"]
        [record] = await _profiled(
            store, store._summaries, **{"command.find": {"$exists": True}}
        )
        assert record["planSummary"].startswith("IXSCAN"), record["planSummary"]
        assert record["docsExamined"] == 3
        # Bookkeeping fields (last_position, rev) stay on the server.
        assert set(record["command"]["projection"]) == {"mtime", "data"}


class TestAppendPlan:
    @pytest.mark.anyio
    async def test_sequential_appends_fold_without_rereading_entries(
        self, make_store: StoreFactory
    ) -> None:
        """In the normal case each append folds only its own batch into the
        summary. Rebuilding the summary from every stored entry is reserved
        for retries and out-of-order appends."""
        store = await make_store()
        key: SessionKey = {"project_key": "proj", "session_id": "s"}
        async with _profiling(store):
            for i in range(5):
                await store.append(key, [{"type": "user", "uuid": f"u{i}"}])

        assert await _profiled(store, store._entries, op="query") == []


class TestDeleteInactivePlan:
    @pytest.mark.anyio
    async def test_sweep_reads_one_counter_per_transcript_and_no_entries(
        self, make_store: StoreFactory
    ) -> None:
        """Finding idle sessions reads the small counters collection (one
        document per transcript), never the entries."""
        n_sessions, n_entries = 5, 200
        store = await make_store()
        for s in range(n_sessions):
            key: SessionKey = {"project_key": "proj", "session_id": f"s{s}"}
            await store.append(
                key, [entry({"type": "user", "n": i}) for i in range(n_entries)]
            )
            await store.append({**key, "subpath": "subagents/a"}, [{"type": "user"}])
        await store._counters.update_many(
            {}, {"$set": {"mtime": int(time.time() * 1000) - 40 * 86_400_000}}
        )

        async with _profiling(store):
            assert await store.delete_inactive(timedelta(days=30)) == n_sessions

        aggregate = {"command.aggregate": {"$exists": True}}
        [record] = await _profiled(store, store._counters, **aggregate)
        assert record["docsExamined"] == 2 * n_sessions
        assert await _profiled(store, store._entries, **aggregate) == []
