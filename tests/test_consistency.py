"""Live tests for ordering, idempotency, and concurrency guarantees.

These cover the correctness fixes from the PR #1014 review.
"""

from __future__ import annotations

import time
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import anyio
import bson.objectid
import pytest
from claude_agent_sdk import SessionKey
from pymongo.errors import DuplicateKeyError

import claude_agent_sdk_mongodb_session_store._store as _store
from claude_agent_sdk_mongodb_session_store import MongoDBSessionStore

from .conftest import StoreFactory

KEY: SessionKey = {"project_key": "proj", "session_id": "sess"}


def _e(uuid: str, **extra: Any) -> dict[str, Any]:
    return {"type": "user", "uuid": uuid, **extra}


def _pause_after_summary_read(
    store: MongoDBSessionStore, monkeypatch: pytest.MonkeyPatch
) -> tuple[anyio.Event, anyio.Event]:
    """Make the store's next summary read stall until released.

    Returns ``(read, release)``: ``read`` is set once the summary has been
    read, and the append resumes when ``release`` is set.
    """
    read, release = anyio.Event(), anyio.Event()
    real_find_one = store._summaries.find_one

    async def find_one(*args: Any, **kwargs: Any) -> Any:
        doc = await real_find_one(*args, **kwargs)
        if not read.is_set():
            read.set()
            await release.wait()
        return doc

    monkeypatch.setattr(store._summaries, "find_one", find_one)
    return read, release


def _pause_before_insert(
    store: MongoDBSessionStore, monkeypatch: pytest.MonkeyPatch
) -> tuple[anyio.Event, anyio.Event]:
    """Make the store's next entry insert stall until released, after the
    append has reserved its positions.

    Returns ``(reserved, release)``: ``reserved`` is set once the append is
    about to insert, and it inserts when ``release`` is set.
    """
    reserved, release = anyio.Event(), anyio.Event()
    real_insert_many = store._entries.insert_many

    async def insert_many(*args: Any, **kwargs: Any) -> Any:
        if not reserved.is_set():
            reserved.set()
            await release.wait()
        return await real_insert_many(*args, **kwargs)

    monkeypatch.setattr(store._entries, "insert_many", insert_many)
    return reserved, release


async def _age_all(store: MongoDBSessionStore, days: int) -> None:
    """Backdate every entry and counter, as if idle for ``days``."""
    aged = {"$set": {"mtime": int(time.time() * 1000) - days * 86_400_000}}
    await store._entries.update_many({}, aged)
    await store._counters.update_many({}, aged)


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
    async def test_delete_keeps_counters(self, make_store: StoreFactory) -> None:
        """A transcript appended to after its delete continues numbering
        rather than starting again at 1, so it sorts after any batch that was
        in flight during the delete."""
        store = await make_store()
        sub: SessionKey = {**KEY, "subpath": "subagents/agent-1"}
        await store.append(KEY, [_e("a")])
        await store.append(sub, [_e("s")])

        await store.delete(sub)
        await store.delete(KEY)
        await store.append(KEY, [_e("b")])
        await store.append(sub, [_e("t")])

        positions = [
            (d["subpath"], d["position"])
            async for d in store._entries.find({}).sort("subpath", 1)
        ]
        assert positions == [("", 2), (sub["subpath"], 2)]
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


class TestDeleteRace:
    @pytest.mark.anyio
    async def test_delete_during_append_leaves_no_summary(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``delete()`` runs while an append has read the summary but not yet
        written it back. The append must not bring the deleted session's
        summary back."""
        store = await make_store()
        await store.append(KEY, [_e("a", customTitle="deleted")])
        read, release = _pause_after_summary_read(store, monkeypatch)

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.append, KEY, [_e("b")])
            await read.wait()
            tg.start_soon(store.delete, KEY)
            # Give the delete time to run, if nothing holds it back.
            await anyio.sleep(0.2)
            release.set()

        assert await store.list_session_summaries("proj") == []
        assert await store.load(KEY) is None

    @pytest.mark.anyio
    async def test_append_racing_delete_keeps_later_appends_after_it(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``delete()`` runs after an append has reserved its positions but
        before it inserts. Its entries survive, as a session holding just
        that batch, and a later append must still come after them."""
        store = await make_store()
        await store.append(KEY, [_e("a"), _e("b"), _e("c")])
        reserved, release = _pause_before_insert(store, monkeypatch)

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.append, KEY, [_e("late")])
            await reserved.wait()
            await store.delete(KEY)
            release.set()
        await store.append(KEY, [_e("next")])

        assert await store.load(KEY) == [_e("late"), _e("next")]

    @pytest.mark.anyio
    async def test_entries_left_by_append_racing_delete_are_swept(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = await make_store()
        await store.append(KEY, [_e("a"), _e("b"), _e("c")])
        reserved, release = _pause_before_insert(store, monkeypatch)

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.append, KEY, [_e("late")])
            await reserved.wait()
            await store.delete(KEY)
            release.set()
        await _age_all(store, 40)
        await store.delete_inactive(timedelta(days=30))

        assert await store.load(KEY) is None
        assert await store.list_session_summaries("proj") == []

    @pytest.mark.anyio
    async def test_summary_left_by_first_append_racing_delete_is_swept(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session's first append folds its batch without reading stored
        entries. A ``delete()`` between its insert and its summary write
        leaves a summary with no entries, which the sweep must remove."""
        store = await make_store()
        read, release = _pause_after_summary_read(store, monkeypatch)

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.append, KEY, [_e("a", customTitle="gone")])
            await read.wait()
            await store.delete(KEY)
            release.set()
        assert await store.load(KEY) is None
        await _age_all(store, 40)
        await store.delete_inactive(timedelta(days=30))

        assert await store.list_session_summaries("proj") == []

    @pytest.mark.anyio
    async def test_subagent_append_racing_its_delete_keeps_order(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = await make_store()
        sub: SessionKey = {**KEY, "subpath": "subagents/agent-1"}
        await store.append(sub, [_e("a"), _e("b"), _e("c")])
        reserved, release = _pause_before_insert(store, monkeypatch)

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.append, sub, [_e("late")])
            await reserved.wait()
            await store.delete(sub)
            release.set()
        await store.append(sub, [_e("next")])

        assert await store.load(sub) == [_e("late"), _e("next")]


class TestAcrossProcesses:
    """Two store instances on the same collections stand in for two
    processes: they share data but not in-process locks."""

    @pytest.mark.anyio
    async def test_concurrent_summary_update_is_not_overwritten(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = await make_store()
        b = await make_store(prefix="c0")
        await a.append(KEY, [_e("first")])
        read, release = _pause_after_summary_read(a, monkeypatch)

        async with anyio.create_task_group() as tg:
            tg.start_soon(a.append, KEY, [_e("t", customTitle="TITLE")])
            await read.wait()
            await b.append(KEY, [_e("g", gitBranch="main")])
            release.set()

        [summary] = await a.list_session_summaries("proj")
        assert summary["data"]["custom_title"] == "TITLE"
        assert summary["data"]["git_branch"] == "main"

    @pytest.mark.anyio
    async def test_delete_in_another_process_leaves_no_summary(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = await make_store()
        b = await make_store(prefix="c0")
        await a.append(KEY, [_e("first", customTitle="deleted")])
        read, release = _pause_after_summary_read(a, monkeypatch)

        async with anyio.create_task_group() as tg:
            tg.start_soon(a.append, KEY, [_e("late")])
            await read.wait()
            await b.delete(KEY)
            release.set()

        assert await a.list_session_summaries("proj") == []
        assert await a.load(KEY) is None

    @pytest.mark.anyio
    async def test_session_recreated_in_another_process_is_not_mixed_with_old(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The session is deleted and started again under the same id while an
        append from before the delete is in flight. The new summary may look
        just like the one the stalled append read, so a version number that
        restarts with the document would not notice. The stale append's
        entry is gone and must not show up in the new summary."""
        a = await make_store()
        b = await make_store(prefix="c0")
        await a.append(KEY, [_e("first")])
        read, release = _pause_after_summary_read(a, monkeypatch)

        async with anyio.create_task_group() as tg:
            tg.start_soon(a.append, KEY, [_e("stale", gitBranch="stale")])
            await read.wait()
            await b.delete(KEY)
            await b.append(KEY, [_e("fresh", customTitle="fresh")])
            release.set()

        assert await a.load(KEY) == [_e("fresh", customTitle="fresh")]
        [summary] = await a.list_session_summaries("proj")
        assert summary["data"]["custom_title"] == "fresh"
        assert "git_branch" not in summary["data"]

    @pytest.mark.anyio
    async def test_gives_up_when_summary_never_settles(
        self, make_store: StoreFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If every write loses the race, ``append()`` raises rather than
        looping forever; the SDK retries the batch."""
        store = await make_store()
        await store.append(KEY, [_e("first")])

        async def always_conflicts(*args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(matched_count=0)

        monkeypatch.setattr(store._summaries, "replace_one", always_conflicts)
        with pytest.raises(RuntimeError, match="summary"):
            await store.append(KEY, [_e("second")])


class TestNoPerSessionState:
    @pytest.mark.anyio
    async def test_store_does_not_grow_with_sessions(
        self, make_store: StoreFactory
    ) -> None:
        """One store instance serves a long-running process across many
        sessions, so it must not keep anything per session in memory."""

        def sizes(store: MongoDBSessionStore) -> dict[str, int]:
            return {
                name: len(value)
                for name, value in vars(store).items()
                if isinstance(value, (dict, list, set))
            }

        store = await make_store()
        before = sizes(store)
        for i in range(20):
            key: SessionKey = {"project_key": "proj", "session_id": f"s{i}"}
            await store.append(key, [_e("a")])
            await store.delete(key)
        assert sizes(store) == before
