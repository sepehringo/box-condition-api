import asyncio
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from box_api.admission import AdmissionGate
from box_api.errors import ApiError
from box_api.scheduler import Scheduler
from box_api.settings import Settings
from tests.test_api import until


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_cancellation_and_close(self):
        gate = AdmissionGate(1)
        first = await gate.acquire(1)
        second = asyncio.create_task(gate.acquire(1))
        third = asyncio.create_task(gate.acquire(1))
        await until(lambda: len(gate.waiters) == 2)
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        first.release()
        lease = await third
        self.assertEqual(gate.in_use, 1)
        waiter = asyncio.create_task(gate.acquire(1))
        await until(lambda: len(gate.waiters) == 1)
        gate.close()
        with self.assertRaises(ApiError):
            await waiter
        lease.release()
        self.assertEqual(gate.in_use, 0)

    async def test_cancellation_after_assignment_does_not_leak_slot(self):
        gate = AdmissionGate(1)
        lease = await gate.acquire(1)
        task = asyncio.create_task(gate.acquire(1))
        await until(lambda: len(gate.waiters) == 1)
        lease.release()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(gate.in_use, 0)
        self.assertEqual(len(gate.waiters), 0)


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.order = []
        self.block = threading.Event()
        self.started = threading.Event()
        self.gate = AdmissionGate(8)
        self.settings = Settings(workers=1, processing_timeout=2)

        def infer(path, confidence):
            self.order.append(Path(path).name)
            self.started.set()
            if len(self.order) == 1 and not self.block.wait(3):
                raise RuntimeError("Test worker did not unblock")
            self.assertTrue(Path(path).exists())
            return {"width": 1, "height": 1, "predictions": []}

        self.scheduler = Scheduler(self.settings, ThreadPoolExecutor(1), infer)
        await self.scheduler.start()

    async def asyncTearDown(self):
        self.block.set()
        await self.scheduler.close()
        self.assertEqual(self.gate.in_use, 0)

    async def job(self, names, budget=1):
        directory = tempfile.TemporaryDirectory()
        paths = []
        for name in names:
            path = Path(directory.name) / name
            path.write_bytes(b"test")
            paths.append(str(path))
        lease = await self.gate.acquire(1)
        job = self.scheduler.submit(paths, names, 0.5, directory, lease, "test", budget)
        lease.release()  # Simulate the HTTP handler releasing its reference.
        return job

    async def test_round_robin_preserves_order_and_worker_bound(self):
        first = await self.job(["a0", "a1", "a2"])
        await until(lambda: self.started.is_set())
        second = await self.job(["b0", "b1"])
        self.assertEqual(self.scheduler.active, 1)
        self.block.set()
        a, b = await asyncio.gather(first.future, second.future)
        self.assertEqual(self.order, ["a0", "a1", "b0", "a2", "b1"])
        self.assertEqual([image["filename"] for image in a], ["a0", "a1", "a2"])
        self.assertEqual([image["filename"] for image in b], ["b0", "b1"])

    async def test_queue_timeout_never_submits_expired_images(self):
        first = await self.job(["active"])
        await until(lambda: self.started.is_set())
        second = await self.job(["expired"], budget=0.03)
        with self.assertRaises(ApiError) as error:
            await second.future
        self.assertEqual(error.exception.status_code, 504)
        await until(lambda: second.finalized)
        self.assertEqual(self.order, ["active"])
        self.block.set()
        await first.future

    async def test_cancelling_job_stops_unscheduled_images(self):
        job = await self.job(["active", "cancelled"])
        await until(lambda: self.started.is_set())
        self.scheduler.cancel(job)
        self.assertTrue(job.future.cancelled())
        self.assertEqual(self.gate.in_use, 1)
        self.block.set()
        await until(lambda: self.gate.in_use == 0)
        self.assertEqual(self.order, ["active"])

    async def test_broken_pool_marks_unready_and_fails_waiting_jobs(self):
        await self.scheduler.close()

        def fail(path, confidence):
            raise BrokenProcessPool("Simulated worker death")

        self.scheduler = Scheduler(self.settings, ThreadPoolExecutor(1), fail)
        await self.scheduler.start()
        first = await self.job(["first"])
        second = await self.job(["second"])
        results = await asyncio.gather(first.future, second.future, return_exceptions=True)
        self.assertTrue(all(isinstance(result, ApiError) for result in results))
        self.assertTrue(all(result.status_code == 500 for result in results))
        self.assertFalse(self.scheduler.ready)
        self.assertEqual(self.scheduler.active, 0)


if __name__ == "__main__":
    unittest.main()
