"""Bounded submission and round-robin scheduling across requests."""
import asyncio
import logging
import multiprocessing
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

from box_api.errors import ApiError, InvalidImage
from box_api.worker import infer_image, initialize_worker, worker_ready

logger = logging.getLogger("uvicorn.error.box_api")


@dataclass(eq=False)
class Job:
    paths: list
    names: list
    confidence: float
    directory: object
    lease: object
    future: object
    request_id: str
    results: list
    next_index: int = 0
    running: int = 0
    stopped: bool = False
    started: float | None = None
    created: float = field(default_factory=time.monotonic)
    admission_wait: float = 0.0
    timer: object = None
    finalized: bool = False


class Scheduler:
    def __init__(self, settings, executor=None, infer=infer_image):
        self.settings = settings
        self.executor = executor
        self.infer = infer
        self.ready = False
        self.closing = False
        self.active = 0
        self.jobs = set()
        self.queue = deque()
        self.tasks = set()
        self.cleanups = set()

    async def start(self):
        if self.executor is None:
            context = multiprocessing.get_context("spawn")
            barrier = context.Barrier(self.settings.workers)
            self.executor = ProcessPoolExecutor(
                max_workers=self.settings.workers, mp_context=context,
                initializer=initialize_worker, initargs=(self.settings, barrier),
            )
            loop = asyncio.get_running_loop()
            probes = [
                loop.run_in_executor(self.executor, worker_ready)
                for _ in range(self.settings.workers)
            ]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*probes), self.settings.startup_timeout
                )
            except BaseException:
                await self.close()
                raise
        self.ready = True

    def submit(self, paths, names, confidence, directory, lease, request_id, wait_budget,
               admission_wait=0.0):
        if not self.ready or self.closing:
            raise ApiError(500, "Inference workers are unavailable")
        if wait_budget <= 0:
            raise ApiError(504, "Queue wait exceeded the configured timeout")
        loop = asyncio.get_running_loop()
        lease.retain()
        job = Job(
            paths=paths, names=names, confidence=confidence,
            directory=directory, lease=lease, future=loop.create_future(),
            request_id=request_id, results=[None] * len(paths), admission_wait=admission_wait,
        )
        # Retrieve exceptions even if the HTTP client has already disconnected.
        job.future.add_done_callback(
            lambda future: future.exception() if not future.cancelled() else None
        )
        self.jobs.add(job)
        self.queue.append(job)
        job.timer = loop.call_later(
            max(0, wait_budget), self.cancel, job,
            ApiError(504, "Queue wait exceeded the configured timeout"),
        )
        self._dispatch()
        return job

    def _dispatch(self):
        if not self.ready or self.closing:
            return
        while self.queue and self.active < self.settings.workers:
            job = self.queue.popleft()
            if job.stopped:
                continue
            if job.started is None:
                job.started = time.monotonic()
                job.timer.cancel()
                job.timer = asyncio.get_running_loop().call_later(
                    self.settings.processing_timeout, self.cancel, job,
                    ApiError(504, "Processing exceeded the configured timeout"),
                )
            index = job.next_index
            job.next_index += 1
            job.running += 1
            self.active += 1
            if job.next_index < len(job.paths):
                self.queue.append(job)
            task = asyncio.create_task(self._run(job, index))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def _run(self, job, index):
        try:
            if job.stopped:
                return
            result = await asyncio.get_running_loop().run_in_executor(
                self.executor, self.infer, job.paths[index], job.confidence
            )
            if not job.stopped:
                job.results[index] = {
                    "index": index, "filename": job.names[index], **result
                }
                if all(result is not None for result in job.results):
                    job.future.set_result(job.results)
                    job.stopped = True
                    job.timer.cancel()
                    logger.info(
                        "request=%s images=%d queue_wait=%.3f processing=%.3f",
                        job.request_id, len(job.paths),
                        job.admission_wait + job.started - job.created,
                        time.monotonic() - job.started,
                    )
        except InvalidImage as exc:
            self.cancel(job, ApiError(422, f"Image {index}: {exc}"))
        except BrokenProcessPool:
            self.ready = False
            logger.exception("Inference process pool failed")
            for pending in tuple(self.jobs):
                self.cancel(pending, ApiError(500, "Inference process pool failed"))
        except Exception:
            logger.exception("Inference failed for request=%s image=%d", job.request_id, index)
            self.cancel(job, ApiError(500, "Inference failed"))
        finally:
            job.running -= 1
            self.active -= 1
            self._maybe_finalize(job)
            self._dispatch()

    def cancel(self, job, error=None):
        if not job.stopped:
            job.stopped = True
            job.timer.cancel()
            self.queue = deque(pending for pending in self.queue if pending is not job)
            if not job.future.done():
                if error:
                    job.future.set_exception(error)
                else:
                    job.future.cancel()
        self._maybe_finalize(job)

    def _maybe_finalize(self, job):
        if job.stopped and job.running == 0 and not job.finalized:
            job.finalized = True
            self.jobs.discard(job)
            task = asyncio.create_task(self._cleanup(job))
            self.cleanups.add(task)
            task.add_done_callback(self.cleanups.discard)

    async def _cleanup(self, job):
        try:
            await asyncio.to_thread(job.directory.cleanup)
        except Exception:
            logger.exception("Temporary upload cleanup failed for request=%s", job.request_id)
        finally:
            job.lease.release()

    async def close(self):
        self.ready = False
        self.closing = True
        for job in tuple(self.jobs):
            self.cancel(job, ApiError(500, "Inference service is shutting down"))
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
        if self.cleanups:
            await asyncio.gather(*tuple(self.cleanups), return_exceptions=True)
        if self.executor is not None:
            await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
