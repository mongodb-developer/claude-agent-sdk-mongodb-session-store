"""Live tests for ordering, idempotency, and concurrency guarantees.

These cover the correctness fixes from the PR #1014 review.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import anyio
import bson.objectid
import pytest
from claude_agent_sdk import SessionKey
from pymongo.errors import DuplicateKeyError

import claude_agent_sdk_mongodb_session_store._store as _store

from .conftest import StoreFactory

KEY: SessionKey = {"project_key": "proj", "session_id": "sess"}


def _e(uuid: str, **extra: Any) -> dict[str, Any]:
    return {"type": "user", "uuid": uuid, **extra}


class TestOrdering:
    @pytest.mark.anyio
    async def test_load_order_ignores_client_objectid_clock(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pymongo generates ``_id`` on the client, from the client's clock.

        A second writer whose clock runs behind produces ObjectIds that sort
        *before* earlier entries, so ordering must not depend on ``_id``.
        """
        store = await make_store()
        await store.append(KEY, [_e("a", n=1), _e("b", n=2)])

        real_time = time.time
        monkeypatch.setattr(
            bson.objectid, "time", SimpleNamespace(time=lambda: real_time() - 3600)
        )
        await store.append(KEY, [_e("c", n=3)])

        assert await store.load(KEY) == [_e("a", n=1), _e("b", n=2), _e("c", n=3)]

    @pytest.mark.anyio
    async def test_interleaved_writers_keep_append_order(
        self, make_store: StoreFactory
    ) -> None:
        """Two store instances (two processes) sharing collections."""
        a = await make_store(prefix="shared")
        b = await make_store(prefix="shared")
        expected = []
        for i in range(10):
            writer = a if i % 2 == 0 else b
            entry = _e(f"u{i}", n=i)
            await writer.append(KEY, [entry])
            expected.append(entry)
        assert await a.load(KEY) == expected
        assert await b.load(KEY) == expected

    @pytest.mark.anyio
    async def test_position_is_unique_per_transcript(
        self, make_store: StoreFactory
    ) -> None:
        """The schema enforces what the counter guarantees: no two entries in
        one transcript share a ``position``. The same ``position`` in another transcript
        is fine."""
        store = await make_store()
        await store.append(KEY, [_e("a")])
        doc = await store._entries.find_one({}, {"_id": 0})
        assert doc is not None
        assert doc["position"] == 1

        with pytest.raises(DuplicateKeyError):
            await store._entries.insert_one(dict(doc))
        await store._entries.insert_one({**doc, "subpath": "subagents/agent-1"})
        await store._entries.insert_one({**doc, "session_id": "other"})

    @pytest.mark.anyio
    async def test_delete_removes_counters(self, make_store: StoreFactory) -> None:
        store = await make_store()
        sub: SessionKey = {**KEY, "subpath": "subagents/agent-1"}
        other: SessionKey = {"project_key": "proj", "session_id": "other"}
        await store.append(KEY, [_e("a")])
        await store.append(sub, [_e("s")])
        await store.append(other, [_e("o")])

        await store.delete(sub)
        remaining = [d["_id"] async for d in store._counters.find({})]
        assert sorted((r["session_id"], r["subpath"]) for r in remaining) == [
            ("other", ""),
            ("sess", ""),
        ]

        await store.delete(KEY)
        remaining = [d["_id"] async for d in store._counters.find({})]
        assert remaining == [
            {"project_key": "proj", "session_id": "other", "subpath": ""}
        ]

        # A fresh transcript under a deleted key starts numbering again.
        await store.append(KEY, [_e("b")])
        assert await store.load(KEY) == [_e("b")]


class TestIdempotency:
    @pytest.mark.anyio
    async def test_retried_batch_is_stored_once(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The SDK retries a failed ``append()`` with the same batch. If the
        entries were stored before the failure, the retry must not store them
        again, and the summary must still reflect the batch."""
        store = await make_store()
        await store.append(KEY, [_e("a", n=1)])
        batch = [_e("b", n=2, customTitle="t"), _e("c", n=3)]

        real_fold = _store.fold_session_summary
        calls = 0

        def fail_once(*args: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("summary write failed")
            return real_fold(*args)

        monkeypatch.setattr(_store, "fold_session_summary", fail_once)
        with pytest.raises(ConnectionError):
            await store.append(KEY, batch)
        await store.append(KEY, batch)

        assert await store.load(KEY) == [_e("a", n=1), *batch]
        [summary] = await store.list_session_summaries("proj")
        assert summary["data"]["custom_title"] == "t"

    @pytest.mark.anyio
    async def test_entries_without_uuid_are_not_deduplicated(
        self, make_store: StoreFactory
    ) -> None:
        """Per the protocol, entries without a ``uuid`` (titles, tags, mode
        markers) are appended every time."""
        store = await make_store()
        tag = {"type": "tag", "tag": "x"}
        await store.append(KEY, [tag])
        await store.append(KEY, [tag])
        assert await store.load(KEY) == [tag, tag]

    @pytest.mark.anyio
    async def test_uuid_is_unique_per_transcript_only(
        self, make_store: StoreFactory
    ) -> None:
        store = await make_store()
        sub: SessionKey = {**KEY, "subpath": "subagents/agent-1"}
        other: SessionKey = {"project_key": "proj", "session_id": "other"}
        for key in (KEY, sub, other, KEY):
            await store.append(key, [_e("a")])
        for key in (KEY, sub, other):
            assert await store.load(key) == [_e("a")]


class TestSummaryFold:
    @pytest.mark.anyio
    async def test_late_fold_does_not_overwrite_newer_summary(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Append A stores its entries, then stalls before folding. Append B
        stores and folds a newer title. A's late fold must not bring the old
        title back, nor move the summary's ``mtime`` behind the entries'."""
        store = await make_store()
        clock = iter(range(1_000, 2_000))
        monkeypatch.setattr(_store, "time", SimpleNamespace(time=lambda: next(clock)))

        a_inserted, release_a = anyio.Event(), anyio.Event()
        real_insert_many = store._entries.insert_many

        async def insert_many(docs: list[dict[str, Any]], **kwargs: Any) -> Any:
            result = await real_insert_many(docs, **kwargs)
            if docs[0]["entry"].get("customTitle") == "old":
                a_inserted.set()
                await release_a.wait()
            return result

        monkeypatch.setattr(store._entries, "insert_many", insert_many)

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.append, KEY, [_e("a", customTitle="old")])
            await a_inserted.wait()
            await store.append(KEY, [_e("b", customTitle="new")])
            release_a.set()

        [summary] = await store.list_session_summaries("proj")
        assert summary["data"]["custom_title"] == "new"
        [listed] = await store.list_sessions("proj")
        assert summary["mtime"] == listed["mtime"]

    @pytest.mark.anyio
    async def test_resent_entry_does_not_roll_back_summary(
        self, make_store: StoreFactory
    ) -> None:
        """An already-stored entry that arrives again is skipped, and must not
        be folded again: re-applying its last-wins fields would roll the
        summary back."""
        store = await make_store()
        await store.append(KEY, [_e("a", customTitle="old")])
        await store.append(KEY, [_e("b", customTitle="new")])
        await store.append(KEY, [_e("a", customTitle="old")])

        [summary] = await store.list_session_summaries("proj")
        assert summary["data"]["custom_title"] == "new"

    @pytest.mark.anyio
    async def test_summary_mtime_never_moves_backward(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A writer whose clock runs behind must not make the summary look
        older than the session. ``list_sessions_from_store()`` treats a
        summary older than ``list_sessions()`` as stale."""
        store = await make_store()
        for clock in (2_000, 1_000):
            monkeypatch.setattr(_store, "time", SimpleNamespace(time=lambda c=clock: c))
            await store.append(KEY, [_e(f"u{clock}")])

        [summary] = await store.list_session_summaries("proj")
        [listed] = await store.list_sessions("proj")
        assert summary["mtime"] == listed["mtime"] == 2_000_000
