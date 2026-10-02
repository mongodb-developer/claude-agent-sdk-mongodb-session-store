"""Live-MongoDB tests for ``MongoDBSessionStore``.

There is no in-process MongoDB mock that faithfully exercises aggregation
and ``distinct``, so these tests are **live-only**: the ``client`` fixture
skips unless ``MONGODB_URI`` is set. Each test uses a random
database name and drops it on teardown.

Run locally::

    docker run -d -p 27017:27017 mongo:latest
    MONGODB_URI=mongodb://localhost:27017 \\
        uv run pytest tests/test_mongodb_session_store.py -v
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    SessionKey,
    SessionStore,
    delete_session_via_store,
    get_session_messages_from_store,
    import_session_to_store,
    list_sessions_from_store,
    project_key_for_directory,
)
from claude_agent_sdk._internal.session_resume import (
    materialize_resume_session,
)
from claude_agent_sdk._internal.transcript_mirror_batcher import (
    TranscriptMirrorBatcher,
)
from claude_agent_sdk.testing import run_session_store_conformance
from pymongo import AsyncMongoClient

from claude_agent_sdk_mongodb_session_store import (
    MongoDBSessionStore,
    MongoDBSessionStoreOptions,
)

from .conftest import entry

SESSION_ID = "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(client: AsyncMongoClient[dict[str, Any]], db_name: str) -> SessionStore:
    s = MongoDBSessionStore(
        options=MongoDBSessionStoreOptions(client=client, db_name=db_name)
    )
    await s.create_schema()
    return s


# ---------------------------------------------------------------------------
# Conformance harness
# ---------------------------------------------------------------------------


class TestConformance:
    @pytest.mark.anyio
    async def test_conformance(
        self, client: AsyncMongoClient[dict[str, Any]], db_name: str
    ) -> None:
        # The harness calls make_store() once per contract for isolation.
        # Give each call its own collection pair so contracts don't see each
        # other's documents; cleanup happens via the db_name teardown.
        counter = itertools.count()

        async def make_store() -> SessionStore:
            n = next(counter)
            s = MongoDBSessionStore(
                client=client,
                db_name=db_name,
                entries_collection=f"entries_{n}",
                summaries_collection=f"summaries_{n}",
                counters_collection=f"counters_{n}",
            )
            await s.create_schema()
            return s

        await run_session_store_conformance(make_store)


# ---------------------------------------------------------------------------
# Adapter-specific invariants the conformance suite cannot probe.
# ---------------------------------------------------------------------------


class TestAdapterSpecific:
    @pytest.mark.anyio
    async def test_create_schema_is_idempotent(
        self, client: AsyncMongoClient[dict[str, Any]], db_name: str
    ) -> None:
        """Calling create_schema() twice must not raise (matches Postgres)."""
        s = MongoDBSessionStore(
            client=client,
            db_name=db_name,
            entries_collection="schema_idem_entries",
            summaries_collection="schema_idem_summaries",
        )
        await s.create_schema()
        await s.create_schema()
        await s.append({"project_key": "p", "session_id": "s"}, [{"type": "a"}])
        loaded = await s.load({"project_key": "p", "session_id": "s"})
        assert loaded == [{"type": "a"}]

    @pytest.mark.anyio
    async def test_options_kwarg_path(
        self, client: AsyncMongoClient[dict[str, Any]], db_name: str
    ) -> None:
        """The dataclass options= path must be equivalent to positional args."""
        s = MongoDBSessionStore(
            options=MongoDBSessionStoreOptions(
                client=client,
                db_name=db_name,
                entries_collection="opts_entries",
                summaries_collection="opts_summaries",
            )
        )
        await s.create_schema()
        await s.append({"project_key": "p", "session_id": "s"}, [{"type": "a"}])
        assert await s.load({"project_key": "p", "session_id": "s"}) == [{"type": "a"}]

    @pytest.mark.anyio
    async def test_subpath_delete_preserves_summary(
        self, client: AsyncMongoClient[dict[str, Any]], db_name: str
    ) -> None:
        """Targeted subpath delete must NOT touch the main session's summary
        sidecar. Only main delete (no subpath) cascades to the summary."""
        s = MongoDBSessionStore(
            client=client,
            db_name=db_name,
            entries_collection="sub_del_entries",
            summaries_collection="sub_del_summaries",
        )
        await s.create_schema()
        key: SessionKey = {"project_key": "p", "session_id": "s"}
        await s.append(key, [entry({"type": "user", "customTitle": "title"})])
        await s.append({**key, "subpath": "subagents/agent-1"}, [{"type": "user"}])
        # Sidecar exists after the main append.
        before = await s.list_session_summaries("p")
        assert len(before) == 1
        # Subpath delete should leave main entries AND the sidecar intact.
        await s.delete({**key, "subpath": "subagents/agent-1"})
        after = await s.list_session_summaries("p")
        assert len(after) == 1
        assert after[0]["data"] == before[0]["data"]
        # And then a main delete actually drops the sidecar.
        await s.delete(key)
        assert await s.list_session_summaries("p") == []

    @pytest.mark.anyio
    async def test_concurrent_appends_serialize_summary_fold(
        self, client: AsyncMongoClient[dict[str, Any]], db_name: str
    ) -> None:
        """Concurrent read-fold-writes must not lose each other's fields.

        Two appends carrying *different* fields (one setting
        ``customTitle``, the other setting ``gitBranch``) can each read
        ``prev=None`` and fold against an empty summary. Without the
        summary's compare-and-swap, the last writer wins entirely and one
        field is clobbered. With it, the loser re-reads and rebuilds, so
        both fields survive.

        Repeating across many trials makes a broken compare-and-swap
        almost certain to produce at least one clobbered run.
        """
        import anyio

        s = MongoDBSessionStore(
            client=client,
            db_name=db_name,
            entries_collection="conc_entries",
            summaries_collection="conc_summaries",
        )
        await s.create_schema()

        for trial in range(30):
            key: SessionKey = {"project_key": "p", "session_id": f"trial-{trial}"}

            # Default-arg binds `key` at definition time so the closures
            # don't capture the mutating loop variable (ruff B023).
            async def with_title(k: SessionKey = key) -> None:
                await s.append(
                    k,
                    [entry({"type": "user", "uuid": "t", "customTitle": "TITLE"})],
                )

            async def with_branch(k: SessionKey = key) -> None:
                await s.append(
                    k, [entry({"type": "user", "uuid": "b", "gitBranch": "main"})]
                )

            async with anyio.create_task_group() as tg:
                tg.start_soon(with_title)
                tg.start_soon(with_branch)

            summaries = [
                s2
                for s2 in await s.list_session_summaries("p")
                if s2["session_id"] == f"trial-{trial}"
            ]
            assert len(summaries) == 1
            data = summaries[0]["data"]
            # Both fields must be present after any interleaving. A missing
            # field => fold raced => regression.
            assert data.get("custom_title") == "TITLE", (
                f"trial {trial}: custom_title clobbered — data={data}"
            )
            assert data.get("git_branch") == "main", (
                f"trial {trial}: git_branch clobbered — data={data}"
            )


# ---------------------------------------------------------------------------
# Full round-trip: TranscriptMirrorBatcher → MongoDB → materialize_resume_session
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.anyio
    async def test_mirror_then_resume(
        self,
        store: SessionStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Isolate ~ so auth-file copying doesn't touch the real config.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

        cwd = tmp_path / "project"
        cwd.mkdir()
        project_key = project_key_for_directory(cwd)

        errors: list[tuple[SessionKey | None, str]] = []

        async def on_error(key: SessionKey | None, msg: str) -> None:
            errors.append((key, msg))

        projects_dir = str(tmp_path / "config" / "projects")
        batcher = TranscriptMirrorBatcher(
            store=store, projects_dir=projects_dir, on_error=on_error
        )

        main_path = f"{projects_dir}/{project_key}/{SESSION_ID}.jsonl"
        sub_path = f"{projects_dir}/{project_key}/{SESSION_ID}/subagents/agent-1.jsonl"
        main_entries = [
            entry(
                {
                    "type": "user",
                    "uuid": "u1",
                    "message": {"role": "user", "content": "hi"},
                }
            ),
            entry(
                {"type": "assistant", "uuid": "a1", "message": {"role": "assistant"}}
            ),
        ]
        sub_entries = [entry({"type": "user", "uuid": "su1", "isSidechain": True})]

        batcher.enqueue(main_path, main_entries)
        batcher.enqueue(sub_path, sub_entries)
        await batcher.flush()
        assert errors == []

        opts = ClaudeAgentOptions(cwd=cwd, session_store=store, resume=SESSION_ID)
        result = await materialize_resume_session(opts)
        assert result is not None
        try:
            assert result.resume_session_id == SESSION_ID
            jsonl = (
                result.config_dir / "projects" / project_key / f"{SESSION_ID}.jsonl"
            ).read_text()
            assert [json.loads(line) for line in jsonl.splitlines()] == main_entries
            sub_jsonl = (
                result.config_dir
                / "projects"
                / project_key
                / SESSION_ID
                / "subagents"
                / "agent-1.jsonl"
            ).read_text()
            assert [json.loads(line) for line in sub_jsonl.splitlines()] == sub_entries
        finally:
            await result.cleanup()

    @pytest.mark.anyio
    async def test_public_api_import_read_list_delete(
        self,
        store: SessionStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The same round trip through the SDK's public helpers only."""
        config = tmp_path / "config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
        cwd = tmp_path / "project"
        cwd.mkdir()
        project_dir = config / "projects" / project_key_for_directory(cwd)
        (project_dir / SESSION_ID / "subagents").mkdir(parents=True)

        main_entries = [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "sessionId": SESSION_ID,
                "timestamp": "2026-01-01T00:00:00.000Z",
                "message": {"role": "user", "content": "hello mongo"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "sessionId": SESSION_ID,
                "timestamp": "2026-01-01T00:00:01.000Z",
                "message": {"role": "assistant", "content": "hi"},
            },
            {
                "type": "custom-title",
                "customTitle": "Imported",
                "sessionId": SESSION_ID,
            },
        ]
        sub_entries = [{"type": "user", "uuid": "su1", "isSidechain": True}]
        (project_dir / f"{SESSION_ID}.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in main_entries)
        )
        (project_dir / SESSION_ID / "subagents" / "agent-1.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in sub_entries)
        )

        # Re-importing must not duplicate anything: uuid is the idempotency key.
        for _ in range(2):
            await import_session_to_store(SESSION_ID, store, directory=str(cwd))

        messages = await get_session_messages_from_store(
            store, SESSION_ID, directory=str(cwd)
        )
        assert [(m.type, m.uuid) for m in messages] == [
            ("user", "u1"),
            ("assistant", "a1"),
        ]

        [info] = await list_sessions_from_store(store, directory=str(cwd))
        assert info.session_id == SESSION_ID
        assert info.custom_title == "Imported"
        assert info.first_prompt == "hello mongo"

        key: SessionKey = {"project_key": project_dir.name, "session_id": SESSION_ID}
        assert await store.load({**key, "subpath": "subagents/agent-1"}) == sub_entries

        await delete_session_via_store(store, SESSION_ID, directory=str(cwd))
        assert await list_sessions_from_store(store, directory=str(cwd)) == []
        assert await store.load(key) is None
        assert await store.load({**key, "subpath": "subagents/agent-1"}) is None


class TestKeyInjection:
    """A non-string key field must never reach the query filter.

    Before validation was added, ``{"$ne": ""}`` as a ``session_id`` acted as
    a query operator: ``load()`` returned every session in the project and
    ``delete()`` removed them all. Seed two sessions in two tenants, attempt
    each operator read and delete, and check nothing leaked or was deleted.
    """

    @pytest.mark.anyio
    async def test_operator_keys_raise_and_leave_data_intact(
        self, store: SessionStore
    ) -> None:
        seeded: dict[tuple[str, str], list[Any]] = {}
        for pk, sid in [("tenant-a", "s1"), ("tenant-a", "s2"), ("tenant-b", "s3")]:
            entries = [entry({"type": "user", "uuid": f"{sid}-u1", "text": sid})]
            await store.append({"project_key": pk, "session_id": sid}, entries)
            seeded[(pk, sid)] = entries

        operators: list[dict[str, Any]] = [
            {"project_key": "tenant-a", "session_id": {"$ne": ""}},
            {"project_key": {"$regex": ".*"}, "session_id": "s3"},
        ]
        for bad in operators:
            with pytest.raises(TypeError):
                await store.load(bad)  # type: ignore[arg-type]
            with pytest.raises(TypeError):
                await store.delete(bad)  # type: ignore[arg-type]
            with pytest.raises(TypeError):
                await store.list_subkeys(bad)  # type: ignore[arg-type]
        bad_subpath: dict[str, Any] = {
            "project_key": "tenant-a",
            "session_id": "s1",
            "subpath": {"$ne": ""},
        }
        with pytest.raises(TypeError):
            await store.load(bad_subpath)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            await store.delete(bad_subpath)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            await store.list_sessions({"$ne": ""})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            await store.list_session_summaries({"$ne": ""})  # type: ignore[arg-type]

        for (pk, sid), entries in seeded.items():
            assert await store.load({"project_key": pk, "session_id": sid}) == entries
        assert sorted(
            s["session_id"] for s in await store.list_sessions("tenant-a")
        ) == [
            "s1",
            "s2",
        ]
