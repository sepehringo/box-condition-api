"""FIFO admission and reference-counted request reservations."""
import asyncio
from collections import deque

from box_api.errors import ApiError


class Lease:
    def __init__(self, gate):
        self.gate = gate
        self.references = 1

    def retain(self):
        if self.references <= 0:
            raise RuntimeError("Cannot retain a released lease")
        self.references += 1

    def release(self):
        if self.references <= 0:
            raise RuntimeError("Lease released twice")
        self.references -= 1
        if self.references == 0:
            self.gate.release()


class AdmissionGate:
    def __init__(self, capacity):
        self.capacity = capacity
        self.in_use = 0
        self.waiters = deque()
        self.closed = False

    async def acquire(self, timeout):
        if self.closed:
            raise ApiError(500, "Inference service is shutting down")
        if self.in_use < self.capacity and not self.waiters:
            self.in_use += 1
            return Lease(self)
        future = asyncio.get_running_loop().create_future()
        self.waiters.append(future)
        task = asyncio.current_task()
        cancellations = task.cancelling()
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout)
            # Python 3.11 wait_for can return a completed future while swallowing
            # simultaneous cancellation. Return its reserved slot in that case.
            if task.cancelling() > cancellations:
                raise asyncio.CancelledError
            return Lease(self)
        except BaseException:
            # A slot may have been assigned just as timeout/cancellation fired.
            if future.done() and not future.cancelled() and future.exception() is None:
                self.release()
            else:
                if future.done() and not future.cancelled():
                    future.exception()
                future.cancel()
            try:
                self.waiters.remove(future)
            except ValueError:
                pass
            raise

    def release(self):
        self.in_use -= 1
        self._wake()

    def _wake(self):
        while self.waiters and self.in_use < self.capacity and not self.closed:
            future = self.waiters.popleft()
            if future.done():
                continue
            self.in_use += 1
            future.set_result(None)

    def close(self):
        self.closed = True
        while self.waiters:
            future = self.waiters.popleft()
            if not future.done():
                future.set_exception(ApiError(500, "Inference service is shutting down"))
