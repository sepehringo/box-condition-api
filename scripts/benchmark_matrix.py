"""Launch temporary CPU servers and benchmark worker/batch/concurrency combinations."""
import argparse
import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from scripts.benchmark_api import benchmark

ROOT = Path(__file__).resolve().parent.parent


async def wait_ready(url, process, log_path):
    deadline = time.monotonic() + 180
    async with httpx.AsyncClient(base_url=url, timeout=1) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(log_path.read_text())
            try:
                response = await client.get("/ready")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError("Server startup exceeded 180 seconds")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", nargs="+", required=True, type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 5, 10, 25])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "api_benchmark_cpu.json")
    args = parser.parse_args()
    if min(*args.workers, *args.concurrency, *args.batch_size, args.requests) <= 0:
        parser.error("Workers, concurrency, batch size and requests must be positive")
    images = [(path, path.read_bytes()) for path in args.image]
    report = {
        "platform": platform.platform(), "python": platform.python_version(),
        "logical_cpus": os.cpu_count(), "torch_threads_per_worker": 1,
        "image_files": [str(path) for path in args.image], "measurements": [],
    }
    for workers in args.workers:
        with tempfile.TemporaryDirectory(prefix="box-benchmark-") as root:
            root = Path(root)
            uploads = root / "uploads"
            uploads.mkdir()
            log_path = root / "server.log"
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            environment = {**os.environ, "BOX_DEVICE": "cpu", "BOX_WORKERS": str(workers),
                           "BOX_TORCH_THREADS": "1", "BOX_TEMP_ROOT": str(uploads)}
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "box_api.app:app", "--host",
                     "127.0.0.1", "--port", str(port), "--workers", "1", "--log-level", "warning"],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    await wait_ready(url, process, log_path)
                    for batch_size in args.batch_size:
                        config = SimpleNamespace(
                            url=url, timeout=150, server_pid=process.pid,
                            temp_root=uploads, batch_size=batch_size, warmup=2,
                            requests=args.requests,
                        )
                        for concurrency in args.concurrency:
                            result = await benchmark(config, concurrency, images)
                            result["workers"] = workers
                            report["measurements"].append(result)
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            args.output.write_text(json.dumps(report, indent=2) + "\n")
                            print(
                                f"workers={workers} batch={batch_size} concurrency={concurrency} "
                                f"images/s={result['successful_images_per_second']} "
                                f"p95={result['success_latency_seconds']['p95']} "
                                f"statuses={result['statuses']}", flush=True,
                            )
                finally:
                    process.terminate()
                    try:
                        await asyncio.to_thread(process.wait, timeout=180)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        await asyncio.to_thread(process.wait)
            if any(uploads.iterdir()):
                raise RuntimeError("Temporary uploads remained after graceful shutdown")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
