import asyncio

import pytest

from bot.__main__ import cancel_and_wait


@pytest.mark.asyncio
async def test_cancel_and_wait_finishes_background_task() -> None:
    cancelled = asyncio.Event()

    async def worker() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(worker())
    await asyncio.sleep(0)

    await cancel_and_wait(task)

    assert task.done()
    assert task.cancelled()
    assert cancelled.is_set()
