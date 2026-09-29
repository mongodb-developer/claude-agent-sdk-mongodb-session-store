"""MongoDB-backed :class:`~claude_agent_sdk.SessionStore`.

Originally contributed as a reference adapter in
anthropics/claude-agent-sdk-python#1014. It mirrors the
``MongoDBSessionStore`` reference implementation from the TypeScript SDK.

Requires ``pymongo>=4.13`` (the stable async API). Install with::

    uv add claude-agent-sdk-mongodb-session-store

Usage::

    from pymongo import AsyncMongoClient
    from claude_agent_sdk import ClaudeAgentOptions, query

    from claude_agent_sdk_mongodb_session_store import MongoDBSessionStore

    client = AsyncMongoClient("mongodb://localhost:27017")
    store = MongoDBSessionStore(client=client, db_name="claude")
    await store.create_schema()  # one-time, idempotent

    async for message in query(
        prompt="Hello!",
        options=ClaudeAgentOptions(session_store=store),
    ):
        ...  # messages are mirrored to MongoDB as they stream

Schema
------
Three collections share a single database:

``claude_session_entries`` — one document per JSONL entry::

    {
      _id: ObjectId,
      project_key: str,
      session_id:  str,
      subpath:     str,                 # "" sentinel for main transcript
      position:    int,                 # append order, from the counter
      uuid:        str,                 # entry["uuid"], when it has one
      entry:       <opaque JSON>,
      mtime:       int,                 # Unix epoch ms, write-time stamp
    }

``claude_session_summaries`` — one document per main session, maintained
incrementally inside :meth:`MongoDBSessionStore.append` via
:func:`~claude_agent_sdk.fold_session_summary`::

    {
      _id:           {project_key: str, session_id: str},
      mtime:         int,               # Unix epoch ms (same clock as entries)
      last_position: int,               # last main-transcript entry folded in
      data:          <opaque SDK-owned dict>,
    }

``claude_session_counters`` — one document per transcript, holding the last
``position`` reserved by :meth:`MongoDBSessionStore.append` and the time of the
latest append::

    {
      _id:           {project_key: str, session_id: str, subpath: str},
      last_position: int,
      mtime:         int,               # Unix epoch ms (same clock as entries)
    }

The empty string is the ``subpath`` sentinel for the main transcript so the
``(project_key, session_id, subpath)`` triple is total (mirrors the Postgres
adapter's convention).

Concurrency
-----------
Per the :meth:`SessionStore.list_session_summaries` contract, stores
maintaining sidecars inside ``append()`` must serialize the read-fold-write
when ``append()`` calls can race for the same session. This adapter holds a
per-session ``anyio.Lock`` keyed by ``(project_key, session_id)`` for the
duration of the summary update. The SDK's own ``TranscriptMirrorBatcher``
already sequences appends per session within one process, but a user could
share one store instance across multiple concurrent batchers — the lock keeps
the fold deterministic in that case.

Retention
---------
This adapter never deletes documents on its own. Schedule
:meth:`MongoDBSessionStore.delete_inactive` to remove sessions that have had
no appends for a given period. It deletes whole sessions only.

Don't expire individual entries (a TTL index, or
``delete_many({"mtime": {"$lt": cutoff}})``): that deletes the oldest entries
of a session that is still in use, leaving a transcript that can't be resumed.
TTL indexes also ignore ``mtime``, which is an integer, not a BSON Date.

Local-disk transcripts under ``CLAUDE_CONFIG_DIR`` are swept independently by
the CLI's ``cleanupPeriodDays`` setting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import anyio
from claude_agent_sdk import (
    SessionKey,
    SessionListSubkeysKey,
    SessionStore,
    SessionStoreEntry,
    SessionStoreListEntry,
    SessionSummaryEntry,
    fold_session_summary,
)
from pymongo import ReturnDocument
from pymongo.errors import BulkWriteError, InvalidName

if TYPE_CHECKING:
    from pymongo import AsyncMongoClient
    from pymongo.asynchronous.collection import AsyncCollection
    from pymongo.asynchronous.database import AsyncDatabase

#: Sentinel used in entry documents to mark the main transcript. The SDK
#: never emits an empty subpath; treating ``key.get("subpath") or ""`` as the
#: sentinel keeps the Mongo query and Postgres adapter aligned.
_MAIN: str = ""


@dataclass
class MongoDBSessionStoreOptions:
    """Configuration for :class:`MongoDBSessionStore`."""

    client: AsyncMongoClient[dict[str, Any]]
    """Pre-configured ``pymongo.AsyncMongoClient``. Caller controls URI,
    auth, TLS, pool sizing, server selection, etc."""

    db_name: str | None = None
    """Database name. Falls back to the client's default database (i.e. the
    one named in the connection URI) when ``None``."""

    entries_collection: str = "claude_session_entries"
    """Collection name for transcript entries. Any name MongoDB accepts,
    except the reserved ``system.`` namespace."""

    summaries_collection: str = "claude_session_summaries"
    """Collection name for the per-session summary sidecar. Any name MongoDB
    accepts, except the reserved ``system.`` namespace."""

    counters_collection: str = "claude_session_counters"
    """Collection name for the per-transcript counters that hand out entry
    positions. Any name MongoDB accepts, except the reserved ``system.``
    namespace."""


class MongoDBSessionStore(SessionStore):
    """MongoDB-backed :class:`~claude_agent_sdk.SessionStore`.

    One document per transcript entry, ordered by a per-transcript
    ``position`` that ``append()`` reserves atomically from a counter
    document. ``load()`` is ``find().sort("position", 1)``.

    Args:
        client: Pre-configured ``pymongo.AsyncMongoClient``.
        db_name: Database name (default: the client's default DB).
        entries_collection: Collection for entry documents
            (default ``"claude_session_entries"``).
        summaries_collection: Collection for summary sidecars
            (default ``"claude_session_summaries"``).
        counters_collection: Collection for position counters
            (default ``"claude_session_counters"``).
        options: Alternative to positional args; takes precedence if given.
    """

    def __init__(
        self,
        client: AsyncMongoClient[dict[str, Any]] | None = None,
        db_name: str | None = None,
        entries_collection: str = "claude_session_entries",
        summaries_collection: str = "claude_session_summaries",
        counters_collection: str = "claude_session_counters",
        *,
        options: MongoDBSessionStoreOptions | None = None,
    ) -> None:
        if options is not None:
            client = options.client
            db_name = options.db_name
            entries_collection = options.entries_collection
            summaries_collection = options.summaries_collection
            counters_collection = options.counters_collection
        if client is None:
            raise ValueError("MongoDBSessionStore requires 'client'")

        self._db: AsyncDatabase[dict[str, Any]] = (
            client[db_name] if db_name is not None else client.get_default_database()
        )
        self._entries = self._collection("entries_collection", entries_collection)
        self._summaries = self._collection("summaries_collection", summaries_collection)
        self._counters = self._collection("counters_collection", counters_collection)
        # Per-session locks for the read-fold-write summary update. Keys are
        # (project_key, session_id); locks are created lazily and never
        # garbage-collected — this is reference code, not a long-running
        # service.
        self._summary_locks: dict[tuple[str, str], anyio.Lock] = {}

    def _collection(self, label: str, name: str) -> AsyncCollection[dict[str, Any]]:
        """Return collection ``name``, raising ``ValueError`` if it is invalid.

        MongoDB collection names are not an injection vector, so validation
        is delegated to pymongo (which rejects empty names, ``$``, null bytes,
        and leading, trailing, or doubled ``.``). The only extra rule is the
        reserved ``system.`` namespace, which pymongo accepts.
        """
        if name.startswith("system."):
            raise ValueError(
                f"{label} {name!r} is invalid: the 'system.' prefix is reserved"
            )
        try:
            return self._db[name]
        except InvalidName as e:
            raise ValueError(f"{label} {name!r} is invalid: {e}") from e

    def _summary_lock(self, key: SessionKey) -> anyio.Lock:
        slot = (key["project_key"], key["session_id"])
        lock = self._summary_locks.get(slot)
        if lock is None:
            lock = anyio.Lock()
            self._summary_locks[slot] = lock
        return lock

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def create_schema(self) -> None:
        """Create the indexes if absent. Idempotent.

        Call once at startup (or run the equivalent migration out-of-band).
        Each index is independently named so re-running the call is a no-op
        in the steady state.
        """
        await self._entries.create_index(
            [("project_key", 1), ("session_id", 1), ("subpath", 1), ("position", 1)],
            name="key_position_idx",
            # The counter already hands out each position once; this enforces it.
            unique=True,
        )
        await self._entries.create_index(
            [("project_key", 1), ("session_id", 1), ("subpath", 1), ("uuid", 1)],
            name="key_uuid_idx",
            unique=True,
            partialFilterExpression={"uuid": {"$type": "string"}},
        )
        # Ordered so list_sessions() can DISTINCT_SCAN: one key per session.
        await self._entries.create_index(
            [("project_key", 1), ("subpath", 1), ("session_id", 1), ("mtime", -1)],
            name="sessions_idx",
        )
        await self._summaries.create_index(
            [("_id.project_key", 1)],
            name="summaries_idx",
        )
        await self._counters.create_index(
            [("_id.project_key", 1), ("_id.session_id", 1)],
            name="counters_session_idx",
        )

    async def _reserve_positions(
        self, key: SessionKey, subpath: str, n: int, mtime: int
    ) -> int:
        """Atomically reserve ``n`` positions and return the first.

        Also records ``mtime`` as the transcript's latest append time, which
        :meth:`delete_inactive` reads.

        The counter is incremented on the server, so reservations are totally
        ordered even across processes with skewed clocks. Ordering by
        client-generated ObjectIds would not be: they sort by the client's
        clock in seconds, then a per-process random value.
        """
        doc = await self._counters.find_one_and_update(
            {
                "_id": {
                    "project_key": key["project_key"],
                    "session_id": key["session_id"],
                    "subpath": subpath,
                }
            },
            {"$inc": {"last_position": n}, "$max": {"mtime": mtime}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        assert doc is not None  # For typing. upsert=True with AFTER never returns None.
        return int(doc["last_position"]) - n + 1

    async def _insert_new(self, docs: list[dict[str, Any]]) -> bool:
        """Insert ``docs``, skipping any whose ``uuid`` is already stored in
        the same transcript. Returns whether none were skipped.

        The SDK retries a failed ``append()`` with the same batch, and treats
        an entry's ``uuid`` as its idempotency key. Unordered, so the entries
        after a duplicate are still inserted; order comes from ``position``.
        """
        try:
            await self._entries.insert_many(docs, ordered=False)
        except BulkWriteError as e:
            errors = e.details["writeErrors"]
            if e.details.get("writeConcernErrors") or not all(
                err["code"] == 11000 and "uuid" in err.get("keyPattern", {})
                for err in errors
            ):
                raise
            return False
        return True

    async def _update_summary(
        self,
        key: SessionKey,
        entries: list[SessionStoreEntry],
        first_position: int,
        mtime: int,
        all_new: bool,
    ) -> None:
        """Fold a main-transcript batch into the session's summary.

        The summary records the last position it has folded in. When this
        batch directly follows it and none of its entries were skipped as
        duplicates, only the batch is folded in, and ``mtime`` only moves
        forward. Otherwise (a retry, a re-sent entry, or appends folding out
        of order) the summary is rebuilt from every stored entry. Folding an
        old entry again would roll its last-wins fields back.
        """
        compound_id = {
            "project_key": key["project_key"],
            "session_id": key["session_id"],
        }
        async with self._summary_lock(key):
            prev_doc = await self._summaries.find_one({"_id": compound_id})
            prev: SessionSummaryEntry | None = (
                {
                    "session_id": prev_doc["_id"]["session_id"],
                    "mtime": int(prev_doc["mtime"]),
                    "data": prev_doc["data"],
                }
                if prev_doc is not None
                else None
            )
            last_position = prev_doc.get("last_position") if prev_doc else 0
            if all_new and last_position == first_position - 1:
                new_doc = {
                    "_id": compound_id,
                    "mtime": max(mtime, prev["mtime"]) if prev else mtime,
                    "last_position": first_position + len(entries) - 1,
                    "data": fold_session_summary(prev, key, entries)["data"],
                }
            else:
                stored = (
                    await self._entries.find(
                        {**compound_id, "subpath": _MAIN},
                        {"entry": 1, "position": 1, "mtime": 1},
                    )
                    .sort("position", 1)
                    .to_list(length=None)
                )
                if not stored:
                    return
                new_doc = {
                    "_id": compound_id,
                    "mtime": max(d["mtime"] for d in stored),
                    "last_position": stored[-1]["position"],
                    "data": fold_session_summary(
                        None, key, [d["entry"] for d in stored]
                    )["data"],
                }
            await self._summaries.replace_one(
                {"_id": compound_id}, new_doc, upsert=True
            )

    # ------------------------------------------------------------------
    # SessionStore protocol
    # ------------------------------------------------------------------

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        if not entries:
            return
        subpath = key.get("subpath") or _MAIN
        now = int(time.time() * 1000)
        first_position = await self._reserve_positions(key, subpath, len(entries), now)
        docs: list[dict[str, Any]] = [
            {
                "project_key": key["project_key"],
                "session_id": key["session_id"],
                "subpath": subpath,
                "position": first_position + i,
                "entry": dict(entry),
                "mtime": now,
            }
            for i, entry in enumerate(entries)
        ]
        for doc in docs:
            if isinstance(doc["entry"].get("uuid"), str):
                doc["uuid"] = doc["entry"]["uuid"]
        all_new = await self._insert_new(docs)
        # Subagent transcripts must NOT contribute to the main session's
        # summary — guard before the fold (per fold_session_summary docs).
        if subpath == _MAIN:
            await self._update_summary(key, entries, first_position, now, all_new)

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        cursor = self._entries.find(
            {
                "project_key": key["project_key"],
                "session_id": key["session_id"],
                "subpath": key.get("subpath") or _MAIN,
            }
        ).sort("position", 1)
        docs = await cursor.to_list(length=None)
        if not docs:
            return None
        return [d["entry"] for d in docs]

    async def list_sessions(self, project_key: str) -> list[SessionStoreListEntry]:
        # Sorting on the sessions_idx key order lets $group/$first read only
        # each session's newest index key (DISTINCT_SCAN), never a document.
        pipeline: list[dict[str, Any]] = [
            {"$match": {"project_key": project_key, "subpath": _MAIN}},
            {"$sort": {"session_id": 1, "mtime": -1}},
            {"$group": {"_id": "$session_id", "mtime": {"$first": "$mtime"}}},
        ]
        return [
            {"session_id": str(r["_id"]), "mtime": int(r["mtime"])}
            async for r in await self._entries.aggregate(pipeline)
        ]

    async def list_session_summaries(
        self, project_key: str
    ) -> list[SessionSummaryEntry]:
        return [
            {
                "session_id": d["_id"]["session_id"],
                "mtime": int(d["mtime"]),
                "data": d["data"],
            }
            async for d in self._summaries.find({"_id.project_key": project_key})
        ]

    async def delete(self, key: SessionKey) -> None:
        subpath = key.get("subpath")
        if subpath:
            # Targeted: remove only this subpath's entries; do NOT touch the
            # summary sidecar (which represents the main transcript).
            await self._entries.delete_many(
                {
                    "project_key": key["project_key"],
                    "session_id": key["session_id"],
                    "subpath": subpath,
                }
            )
            await self._counters.delete_one(
                {
                    "_id": {
                        "project_key": key["project_key"],
                        "session_id": key["session_id"],
                        "subpath": subpath,
                    }
                }
            )
            return
        # Cascade: main + every subpath under (project_key, session_id),
        # their position counters, and the summary sidecar.
        await self._entries.delete_many(
            {
                "project_key": key["project_key"],
                "session_id": key["session_id"],
            }
        )
        await self._counters.delete_many(
            {
                "_id.project_key": key["project_key"],
                "_id.session_id": key["session_id"],
            }
        )
        await self._summaries.delete_one(
            {
                "_id": {
                    "project_key": key["project_key"],
                    "session_id": key["session_id"],
                }
            }
        )

    async def list_subkeys(self, key: SessionListSubkeysKey) -> list[str]:
        result = await self._entries.distinct(
            "subpath",
            {
                "project_key": key["project_key"],
                "session_id": key["session_id"],
                "subpath": {"$ne": _MAIN},
            },
        )
        return list(result)

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    async def delete_inactive(self, older_than: timedelta) -> int:
        """Delete every session with no appends for ``older_than``.

        A session counts as active if any of its transcripts, main or
        subagent, was appended to since the cutoff. Inactive sessions are
        removed whole via :meth:`delete`, so a long-running session is never
        truncated. Covers every project. Returns the number of sessions
        deleted.

        Run it on a schedule (cron, an Atlas scheduled trigger, ...). A
        session appended to at the moment the sweep deletes it can still be
        lost, having been idle for ``older_than`` until then.
        """
        if older_than <= timedelta(0):
            raise ValueError(f"older_than must be positive, got {older_than!r}")
        cutoff = int(time.time() * 1000) - older_than // timedelta(milliseconds=1)
        pipeline: list[dict[str, Any]] = [
            {
                "$group": {
                    "_id": {
                        "project_key": "$_id.project_key",
                        "session_id": "$_id.session_id",
                    },
                    "mtime": {"$max": "$mtime"},
                }
            },
            {"$match": {"mtime": {"$lt": cutoff}}},
        ]
        idle = [r["_id"] async for r in await self._counters.aggregate(pipeline)]
        for session in idle:
            await self.delete(
                {
                    "project_key": session["project_key"],
                    "session_id": session["session_id"],
                }
            )
        return len(idle)
