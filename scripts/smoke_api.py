"""Check a live or containerized API using only Python's standard library."""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def request(url, path, data=None, headers=None):
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as response:
        return response.code, response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    key = os.environ["BOX_API_KEY"]
    deadline = time.monotonic() + 240
    while True:
        try:
            status, _ = request(args.url, "/ready")
            if status == 200:
                break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("API did not become ready")
        time.sleep(1)
    for path in ("/health", "/docs", "/openapi.json"):
        status, _ = request(args.url, path)
        assert status == 200, (path, status)
    for supplied in (None, "incorrect-key"):
        status, _ = request(args.url, "/v1/predict", b"unparsed-body",
                            {"X-API-Key": supplied} if supplied else {})
        assert status == 401, status
    paths = [ROOT / "samples" / name for name in ("box-1.jpg", "box-2.jpg")]
    boundary = "box-condition-smoke-boundary"
    body = b"".join(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
         f'filename="{path.name}"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
        + path.read_bytes() + b"\r\n" for path in paths
    ) + f"--{boundary}--\r\n".encode()
    headers = {"X-API-Key": key, "Content-Type": f"multipart/form-data; boundary={boundary}"}

    def predict(_):
        status, raw = request(args.url, "/v1/predict", body, headers)
        assert status == 200, (status, raw.decode())
        result = json.loads(raw)
        assert [item["index"] for item in result["results"]] == [0, 1]
        assert [item["filename"] for item in result["results"]] == [path.name for path in paths]
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(predict, range(4)))
    if args.output:
        args.output.write_text(json.dumps(results[0], indent=2) + "\n")
    print("Public docs/health, authentication, and four concurrent two-image batches passed.")


if __name__ == "__main__":
    main()
