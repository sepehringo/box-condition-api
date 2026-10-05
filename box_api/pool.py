"""Python 3.11 process-pool adapter for bounded exceptional shutdown.

3.11 has no public terminate_workers API. Keep its private process/thread lookup
here; process termination itself uses multiprocessing's public methods. Real
spawned-process regression tests protect this compatibility boundary.
"""
import time
from concurrent.futures import ProcessPoolExecutor


class InferencePool(ProcessPoolExecutor):
    def abort(self, budget=5.0):
        deadline = time.monotonic() + budget
        with self._shutdown_lock:
            processes = tuple((self._processes or {}).values())
            manager = self._executor_manager_thread
        for process in processes:
            if process.is_alive():
                process.terminate()
        soft_deadline = min(deadline, time.monotonic() + budget / 2)
        for process in processes:
            process.join(max(0, soft_deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        self.shutdown(wait=False, cancel_futures=True)
        if manager is not None:
            manager.join(max(0, deadline - time.monotonic()))
        if any(process.is_alive() for process in processes) or (
            manager is not None and manager.is_alive()
        ):
            raise TimeoutError("Inference pool could not be reaped within cleanup budget")
