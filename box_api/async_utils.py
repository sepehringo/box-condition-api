"""Finish owned background work before propagating handler cancellation."""
import asyncio


async def finish_task(task):
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            if not cancelled:
                raise
            break
    if cancelled:
        # Retrieve failures even when cancellation takes precedence.
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def complete_io(function, *args):
    return await finish_task(asyncio.create_task(asyncio.to_thread(function, *args)))
