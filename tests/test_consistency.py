"""Live tests for ordering, idempotency, and concurrency guarantees.

These cover the correctness fixes from the PR #1014 review.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import bson.objectid
import pytest
from claude_agent_sdk import SessionKey

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
