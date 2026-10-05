import asyncio
import inspect
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from box_api.admission import AdmissionGate
from box_api.app import complete_io
from box_api.pool import InferencePool
from box_api.scheduler import Scheduler
from box_api.settings import Settings
from tests.test_api import until


def stuck_initializer(settings, barrier):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (Path(settings.temp_root) / "started").touch()
    while True:
        time.sleep(1)


def stuck_inference(path, confidence):
    Path(path + ".started").touch()
    while True:
        time.sleep(1)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_finishes_write_before_closing_file(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload"

            async def handler():
                try:
                    with path.open("wb") as output:
                        def write():
                            started.set()
                            if not release.wait(3):
                                raise TimeoutError("Test write did not unblock")
                            output.write(b"complete")
                            finished.set()
                        await complete_io(write)
                finally:
                    path.unlink()

            task = asyncio.create_task(handler())
            try:
                await until(started.is_set)
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0.01)
                    self.assertFalse(task.done())
                    self.assertTrue(path.exists())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(finished.is_set())
                self.assertFalse(path.exists())
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_stuck_startup_is_killed_reaped_and_close_is_idempotent(self):
        before = {child.pid for child in multiprocessing.active_children()}
        with tempfile.TemporaryDirectory() as directory:
            scheduler = Scheduler(Settings(workers=1, startup_timeout=0.8, temp_root=directory))
            start = time.monotonic()
            with patch("box_api.scheduler.initialize_worker", stuck_initializer):
                with self.assertRaises(asyncio.TimeoutError):
                    await scheduler.start()
            self.assertTrue((Path(directory) / "started").exists())
            self.assertLess(time.monotonic() - start, 6)
            self.assertFalse(scheduler.ready)
            await scheduler.close()
            await scheduler.close()
            self.assertFalse({child.pid for child in multiprocessing.active_children()} - before)

    async def test_shutdown_deadline_stops_worker_before_upload_cleanup(self):
        before = {child.pid for child in multiprocessing.active_children()}
        pool = InferencePool(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        scheduler = Scheduler(Settings(workers=1, shutdown_timeout=0.05), pool, stuck_inference)
        await scheduler.start()
        gate = AdmissionGate(1)
        directory = tempfile.TemporaryDirectory()
        path = str(Path(directory.name) / "image")
        Path(path).write_bytes(b"upload")
        lease = await gate.acquire(1)
        job = scheduler.submit([path], ["image"], 0.5, directory, lease, "shutdown", 10)
        lease.release()
        try:
            await until(lambda: Path(path + ".started").exists(), timeout=5)
            self.assertEqual(gate.in_use, 1)
            await asyncio.wait_for(scheduler.close(), 6)
            self.assertTrue(job.future.done())
            self.assertFalse(Path(path).exists())
            self.assertEqual(gate.in_use, 0)
            self.assertFalse({child.pid for child in multiprocessing.active_children()} - before)
        finally:
            await scheduler.close()
            directory.cleanup()


class ConfigurationTests(unittest.TestCase):
    def test_all_timeout_settings_reject_nonfinite_and_nonpositive_values(self):
        for name in ("queue_timeout", "processing_timeout", "upload_timeout",
                     "startup_timeout", "shutdown_timeout"):
            for value in (float("nan"), float("inf"), float("-inf"), 0, -1):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Settings(**{name: value})
                with patch.dict(os.environ, {"BOX_" + name.upper(): str(value)}, clear=True):
                    with self.assertRaises(ValueError):
                        Settings.from_env()
        self.assertEqual(Settings.from_env().shutdown_timeout, 30)

    def test_cli_and_model_loader_default_to_cpu(self):
        import inference
        self.assertEqual(inspect.signature(inference.load_model).parameters["device"].default, "cpu")
        with patch("sys.argv", ["inference.py", "--image", "example.jpg"]), \
             patch.object(inference, "load_model", return_value=Mock()) as load, \
             patch.object(inference, "predict_single_image", return_value=None) as predict:
            inference.main()
        self.assertEqual(load.call_args.args[1], "cpu")
        self.assertEqual(predict.call_args.kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
