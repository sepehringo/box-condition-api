#!/usr/bin/env python3
"""Measure concurrent HTTP inference; optionally sample server memory and queue state."""
import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import httpx


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return round(values[max(0, math.ceil(len(values) * fraction) - 1)], 4)


async def benchmark(args, concurrency, images):
    statuses = Counter()
    success_latencies, all_latencies = [], []
    resources = {
        "peak_server_rss_mib": None, "peak_admitted_requests": 0,
        "peak_waiting_admission": 0, "peak_active_workers": 0,
        "peak_temp_disk_mib": None,
    }
    stop = asyncio.Event()
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 1, max_keepalive_connections=concurrency + 1)

    async with httpx.AsyncClient(
        base_url=args.url, timeout=args.timeout, limits=limits,
        headers={"X-API-Key": os.environ["BOX_API_KEY"]} if os.getenv("BOX_API_KEY") else {},
    ) as client:
        response = await client.get("/ready")
        response.raise_for_status()

        async def sample():
            while not stop.is_set():
                try:
                    state = (await client.get("/ready")).json()
                    for name in ("admitted_requests", "waiting_admission", "active_workers"):
                        resources[f"peak_{name}"] = max(resources[f"peak_{name}"], state.get(name, 0))
                    if args.server_pid:
                        import psutil
                        process = psutil.Process(args.server_pid)
                        processes = [process, *process.children(recursive=True)]
                        rss = 0
                        for child in processes:
                            try:
                                rss += child.memory_info().rss
                            except psutil.Error:
                                pass
                        resources["peak_server_rss_mib"] = max(
                            resources["peak_server_rss_mib"] or 0, rss / 1024 ** 2
                        )
                    if args.temp_root:
                        disk = sum(
                            path.stat().st_size for path in Path(args.temp_root).rglob("*")
                            if path.is_file()
                        )
                        resources["peak_temp_disk_mib"] = max(
                            resources["peak_temp_disk_mib"] or 0, disk / 1024 ** 2
                        )
                except (httpx.HTTPError, OSError, ValueError):
                    pass
                try:
                    await asyncio.wait_for(stop.wait(), 0.2)
                except asyncio.TimeoutError:
                    pass

        async def post(index, measure=True):
            async with semaphore:
                batch = [
                    images[(index * args.batch_size + offset) % len(images)]
                    for offset in range(args.batch_size)
                ]
                started = time.perf_counter()
                try:
                    response = await client.post(
                        "/v1/predict",
                        files=[("files", (path.name, data, "application/octet-stream"))
                               for path, data in batch],
                    )
                    status = str(response.status_code)
                except httpx.HTTPError as exc:
                    status = type(exc).__name__
                latency = time.perf_counter() - started
                if measure:
                    statuses[status] += 1
                    all_latencies.append(latency)
                    if status == "200":
                        success_latencies.append(latency)

        await asyncio.gather(*(post(index, False) for index in range(args.warmup)))
        sampler = asyncio.create_task(sample())
        started = time.perf_counter()
        try:
            await asyncio.gather(*(post(index) for index in range(args.requests)))
        finally:
            elapsed = time.perf_counter() - started
            stop.set()
            await sampler

    return {
        "concurrency": concurrency, "batch_size": args.batch_size,
        "requests": args.requests, "elapsed_seconds": round(elapsed, 3),
        "successful_images_per_second": round(
            len(success_latencies) * args.batch_size / elapsed, 3
        ),
        "success_latency_seconds": {
            "p50": percentile(success_latencies, 0.5),
            "p95": percentile(success_latencies, 0.95),
            "p99": percentile(success_latencies, 0.99),
        },
        "all_latency_seconds": {
            "p50": percentile(all_latencies, 0.5),
            "p95": percentile(all_latencies, 0.95),
        },
        "statuses": dict(statuses),
        "timeout_rate": round(
            sum(count for status, count in statuses.items()
                if status == "504" or "Timeout" in status) / args.requests, 4
        ),
        **resources,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", nargs="+", required=True, type=Path)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 5, 10, 25])
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.concurrency, args.requests, args.batch_size, args.timeout) <= 0:
        parser.error("Concurrency, requests, batch size, and timeout must be positive")
    if args.warmup < 0:
        parser.error("Warmup must be nonnegative")
    images = [(path, path.read_bytes()) for path in args.image]
    results = [
        await benchmark(args, concurrency, images)
        for concurrency in args.concurrency
    ]
    output = json.dumps(results, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    asyncio.run(main())
