<!-- MongoDB sections extracted from examples/session_stores/README.md
     in anthropics/claude-agent-sdk-python#1014. -->

## Production checklist

### MongoDB

- Size the `AsyncMongoClient` connection pool for expected concurrent
  sessions; don't share a pool with request-handler code that holds
  connections.
- The summary sidecar is read-fold-written inside `append()`. The adapter
  serializes per-session updates with an `anyio.Lock`, but multi-process
  writers against the same session would still race — pin a session to a
  single writer or layer your own coordination on top.
- Implement retention via a TTL index on `mtime` or a scheduled
  `delete_many` — both collections grow unbounded.

---

## MongoDB — `mongodb_session_store.py`

Backed by the official [`pymongo`](https://www.mongodb.com/docs/drivers/pymongo/)
driver via its stable async API (`pymongo.AsyncMongoClient`, introduced in
pymongo 4.13).

### Installation

```bash
pip install claude-agent-sdk 'pymongo>=4.13'
```

### Usage

```python
from pymongo import AsyncMongoClient
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from mongodb_session_store import MongoDBSessionStore

client = AsyncMongoClient("mongodb://localhost:27017")
store = MongoDBSessionStore(client=client, db_name="claude")
await store.create_schema()  # idempotent createIndexes

async for message in query(
    prompt="Hello!",
    options=ClaudeAgentOptions(session_store=store),
):
    if isinstance(message, ResultMessage) and message.subtype == "success":
        print(message.result)
```

### Schema

Two collections share a single database. Entries — one document per
transcript entry, ordered by the server-assigned `_id` (`ObjectId`):

```python
{
  "_id": ObjectId,
  "project_key": str,
  "session_id":  str,
  "subpath":     str,    # "" sentinel for the main transcript
  "entry":       <opaque JSON>,
  "mtime":       int,    # Unix epoch ms, write-time stamp
}
```

Summaries — one document per main session, maintained incrementally inside
`append()` via `fold_session_summary`:

```python
{
  "_id":   {"project_key": str, "session_id": str},
  "mtime": int,           # epoch ms (same clock as entries)
  "data":  <opaque>,      # SDK-owned summary state, persisted verbatim
}
```

`create_schema()` creates three indexes — `(project_key, session_id, subpath,
_id)` covering `load()`/`delete()`/`list_subkeys()`, `(project_key, subpath,
mtime DESC)` covering `list_sessions()`, and `(_id.project_key, mtime DESC)`
on the summaries collection. `append()` is a single `insert_many` plus an
atomic summary upsert; `load()` is `find().sort("_id", 1)`.

### Why a summary sidecar?

Unlike the S3, Redis, and Postgres reference adapters in this directory,
this adapter implements the optional `list_session_summaries` method.
`list_sessions_from_store()` takes a fast path when a store offers it —
**one** batch read for all summaries plus a cheap `list_sessions()` to
gap-fill stale or missing entries — instead of falling back to **N**
per-session `load()` calls (bounded at 16 concurrent) to recompute the
summary on the fly. For projects with many sessions or remote-backend
latency, that's the difference between one round trip and dozens.

The summary itself is computed by the SDK's
`fold_session_summary` (read inside `append()` and written back as one
opaque `data` blob); the adapter never interprets the contents. Note
that `fold_session_summary` lives under the SDK's `_internal` package —
treat it as a public surface for adapters specifically (the
`SessionStore` protocol references it) but keep the import in one
place so a future relocation is a single-line fix.

### Concurrency

Per the `SessionStore.list_session_summaries` contract, sidecar updates
inside `append()` must be serialized when calls can race for the same
session. The adapter holds a per-session `anyio.Lock` keyed by
`(project_key, session_id)` for the duration of the read-fold-write.

### Retention

This adapter never deletes documents on its own. Add a TTL index on
`mtime` (the entries collection's `mtime` is epoch ms; convert to seconds
or use a `Date` field instead) or schedule a `delete_many({"mtime": {"$lt":
cutoff}})` to expire transcripts according to your compliance requirements.

`delete()` is implemented (cascades to subpath documents and the summary
sidecar) but is only called when you invoke `delete_session_via_store()`
from the SDK.

Local-disk transcripts under `CLAUDE_CONFIG_DIR` are swept independently by
the CLI's `cleanupPeriodDays` setting.

### Resume from MongoDB

```python
async for message in query(
    prompt="Continue where we left off",
    options=ClaudeAgentOptions(
        session_store=store,
        resume="previous-session-id",
    ),
):
    ...
```

### Live MongoDB end-to-end

There is no in-process MongoDB mock that faithfully exercises aggregation
and `distinct`, so the MongoDB tests run **live-only**. They skip
automatically unless `SESSION_STORE_MONGODB_URL` is set:

```bash
docker run -d -p 27017:27017 mongo:latest

SESSION_STORE_MONGODB_URL=mongodb://localhost:27017 \
    pytest tests/test_example_mongodb_session_store.py -v
```

Each run uses a random database name and drops it on teardown.

This mirrors the `MongoDBSessionStore` reference adapter in the TypeScript
SDK's [`examples/session-stores/mongodb/`](https://github.com/anthropics/claude-agent-sdk-typescript/tree/main/examples/session-stores/mongodb).
