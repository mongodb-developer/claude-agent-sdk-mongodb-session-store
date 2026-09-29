"""Live tests pinning the query plans the adapter relies on for performance.

Each test runs the real store method with the database profiler on, then reads
the server's own record of how the command executed.
"""

from __future__ import annotations

from typing import Any

import pytest
from claude_agent_sdk import SessionKey

from .conftest import StoreFactory


async def _profiled(store: Any, command: str) -> dict[str, Any]:
    """Return the profiler record of the last ``command`` on the entries
    collection."""
    records = (
        await store._db["system.profile"]
        .find(
            {
                "ns": f"{store._db.name}.{store._entries.name}",
                f"command.{command}": {"$exists": True},
            }
        )
        .sort("ts", -1)
        .limit(1)
        .to_list()
    )
    assert records, f"no profiled {command} on {store._entries.name}"
    return dict(records[0])


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
                key, [{"type": "user", "n": i} for i in range(n_entries)]
            )
            # Subagent entries must neither be listed nor examined.
            await store.append(
                {**key, "subpath": "subagents/a"}, [{"type": "user"}] * 50
            )

        await store._db.command("profile", 2)
        try:
            listed = await store.list_sessions("proj")
        finally:
            await store._db.command("profile", 0)

        assert sorted(e["session_id"] for e in listed) == [
            f"s{s}" for s in range(n_sessions)
        ]
        record = await _profiled(store, "aggregate")
        assert record["planSummary"].startswith("DISTINCT_SCAN"), record["planSummary"]
        assert record["docsExamined"] == 0
        # A few keys per session at most, never one per entry.
        assert record["keysExamined"] <= 2 * n_sessions + 1, record["keysExamined"]
