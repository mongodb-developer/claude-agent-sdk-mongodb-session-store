"""Live tests for entries too large for a MongoDB document (16 MiB).

The server and driver reject an oversized document in different ways
depending on how far over the limit it is: a ``BulkWriteError``, an
``OperationFailure`` with code 10334, or the driver's ``DocumentTooLarge``.
``append()`` reports all of them as ``EntryTooLargeError``.
"""

from __future__ import annotations

import pytest
from claude_agent_sdk import SessionKey
from pymongo.errors import DocumentTooLarge

from claude_agent_sdk_mongodb_session_store import EntryTooLargeError

from .conftest import StoreFactory, entry

KEY: SessionKey = {"project_key": "proj", "session_id": "sess"}

_LIMIT = 16 * 1024 * 1024
_KIB = 1024
_MIB = 1024 * 1024


def _big(uuid: str, size: int) -> object:
    return entry({"type": "user", "uuid": uuid, "blob": "x" * size})


class TestEntryTooLarge:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "over", [_KIB, 20 * _KIB, _MIB, 64 * _MIB], ids=lambda n: f"{n}B-over"
    )
    async def test_oversized_entry_raises_specific_error(
        self, make_store: StoreFactory, over: int
    ) -> None:
        store = await make_store()
        with pytest.raises(EntryTooLargeError) as info:
            await store.append(KEY, [_big("big-1", _LIMIT + over)])  # type: ignore[list-item]
        message = str(info.value)
        assert "big-1" in message
        assert "proj" in message and "sess" in message
        assert "16 MiB" in message
        assert info.value.__cause__ is not None

    @pytest.mark.anyio
    async def test_error_is_a_document_too_large(
        self, make_store: StoreFactory
    ) -> None:
        store = await make_store()
        with pytest.raises(DocumentTooLarge):
            await store.append(KEY, [_big("big-1", _LIMIT + _MIB)])  # type: ignore[list-item]

    @pytest.mark.anyio
    async def test_names_the_oversized_entry_in_a_mixed_batch(
        self, make_store: StoreFactory
    ) -> None:
        store = await make_store()
        batch = [
            entry({"type": "user", "uuid": "small-1"}),
            _big("big-2", _LIMIT + _MIB),
            entry({"type": "user", "uuid": "small-3"}),
        ]
        with pytest.raises(EntryTooLargeError) as info:
            await store.append(KEY, batch)  # type: ignore[arg-type]
        assert "big-2" in str(info.value)
        assert "small-1" not in str(info.value)

    @pytest.mark.anyio
    async def test_store_still_works_after_the_failure(
        self, make_store: StoreFactory
    ) -> None:
        store = await make_store()
        with pytest.raises(EntryTooLargeError):
            await store.append(KEY, [_big("big-1", _LIMIT + _MIB)])  # type: ignore[list-item]
        ok = entry({"type": "user", "uuid": "ok-1"})
        await store.append(KEY, [ok])
        assert await store.load(KEY) == [ok]

    @pytest.mark.anyio
    async def test_large_but_valid_entry_is_stored(
        self, make_store: StoreFactory
    ) -> None:
        store = await make_store()
        fits = _big("fits-1", _LIMIT - _MIB)
        await store.append(KEY, [fits])  # type: ignore[list-item]
        assert await store.load(KEY) == [fits]
