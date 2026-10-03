import asyncio
import io
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from box_api.app import create_app
from box_api.scheduler import Scheduler
from box_api.settings import Settings


def image_bytes(width=12, height=10):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


async def until(predicate, timeout=2):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(poll(), timeout)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.block = threading.Event()
        self.block.set()
        self.started = threading.Event()
        self.paths = []
        settings = Settings(
            workers=1, max_requests=2, queue_timeout=0.15,
            processing_timeout=1, temp_root=self.root.name,
        )
        await self.open_app(settings)

    async def open_app(self, settings):
        def infer(path, confidence):
            self.paths.append(path)
            self.started.set()
            if not self.block.wait(3):
                raise RuntimeError("Test worker did not unblock")
            with Image.open(path) as image:
                return {"width": image.width, "height": image.height, "predictions": []}

        self.app = create_app(
            settings,
            lambda config: Scheduler(config, ThreadPoolExecutor(config.workers), infer),
        )
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )

    async def reopen(self, **changes):
        settings = replace(self.app.state.settings, **changes)
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        await self.open_app(settings)

    async def asyncTearDown(self):
        self.block.set()
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.root.cleanup()

    async def predict(self, images=None, **kwargs):
        images = images or [image_bytes()]
        return await self.client.post(
            "/v1/predict",
            files=[("files", ("same.png", data, "image/png")) for data in images],
            **kwargs,
        )

    async def test_batch_order_duplicate_names_and_empty_detections(self):
        response = await self.predict([image_bytes(12), image_bytes(20)])
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()["results"]
        self.assertEqual([item["index"] for item in results], [0, 1])
        self.assertEqual([item["filename"] for item in results], ["same.png", "same.png"])
        self.assertEqual([item["width"] for item in results], [12, 20])
        self.assertEqual([item["predictions"] for item in results], [[], []])
        await until(lambda: self.app.state.gate.in_use == 0)
        self.assertEqual(list(Path(self.root.name).iterdir()), [])

    async def test_invalid_upload_and_confidence(self):
        for data in (b"broken", b""):
            response = await self.predict([data])
            self.assertEqual(response.status_code, 422, response.text)
        for confidence in ("nan", "2", "bad"):
            response = await self.predict(data={"confidence": confidence})
            self.assertEqual(response.status_code, 422, response.text)
        response = await self.client.post("/v1/predict", json={"files": []})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.app.state.gate.in_use, 0)
        self.assertEqual(list(Path(self.root.name).iterdir()), [])

    async def test_upload_byte_and_pixel_limits(self):
        await self.reopen(max_file_bytes=8)
        self.assertEqual((await self.predict()).status_code, 413)
        await self.reopen(max_file_bytes=1024, max_pixels=10)
        self.assertEqual((await self.predict()).status_code, 422)
        await self.reopen(max_body_bytes=8)
        self.assertEqual((await self.predict()).status_code, 413)
        self.assertEqual(self.app.state.gate.in_use, 0)

    async def test_empty_batch_extra_fields_and_file_count(self):
        response = await self.client.post("/v1/predict", data={"confidence": "0.5"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual((await self.predict(data={"other": "x"})).status_code, 422)
        await self.reopen(max_images=1)
        self.assertEqual((await self.predict([image_bytes(), image_bytes()])).status_code, 422)

    async def test_waits_at_capacity_without_reading_second_upload(self):
        await self.reopen(max_requests=1, queue_timeout=1)
        self.block.clear()
        first = asyncio.create_task(self.predict())
        await until(lambda: self.started.is_set())
        consumed = asyncio.Event()
        data = image_bytes()
        body = (
            b"--BOUNDARY\r\nContent-Disposition: form-data; name=\"files\"; "
            b"filename=\"second.png\"\r\nContent-Type: image/png\r\n\r\n"
            + data + b"\r\n--BOUNDARY--\r\n"
        )

        async def stream():
            consumed.set()
            yield body

        second = asyncio.create_task(self.client.post(
            "/v1/predict", content=stream(),
            headers={"Content-Type": "multipart/form-data; boundary=BOUNDARY"},
        ))
        await until(lambda: len(self.app.state.gate.waiters) == 1)
        self.assertFalse(consumed.is_set(), "Upload was consumed before admission")
        health = await asyncio.wait_for(self.client.get("/health"), 0.2)
        self.assertEqual(health.status_code, 200)
        self.block.set()
        results = await asyncio.gather(first, second)
        self.assertEqual([response.status_code for response in results], [200, 200])
        self.assertTrue(consumed.is_set())

    async def test_admission_timeout_returns_504_and_cleans_waiter(self):
        await self.reopen(max_requests=1)
        self.block.clear()
        first = asyncio.create_task(self.predict())
        await until(lambda: self.started.is_set())
        second = await self.predict()
        self.assertEqual(second.status_code, 504, second.text)
        self.assertEqual(len(self.app.state.gate.waiters), 0)
        self.assertEqual(self.app.state.gate.in_use, 1)
        self.block.set()
        self.assertEqual((await first).status_code, 200)

    async def test_processing_timeout_retains_files_and_reservation(self):
        await self.reopen(processing_timeout=0.05)
        self.block.clear()
        response = await self.predict()
        self.assertEqual(response.status_code, 504, response.text)
        self.assertEqual(self.app.state.gate.in_use, 1)
        self.assertTrue(Path(self.paths[0]).exists())
        self.block.set()
        await until(lambda: self.app.state.gate.in_use == 0)
        self.assertFalse(Path(self.paths[0]).exists())

    async def test_cancelled_handler_retains_running_work_until_completion(self):
        self.block.clear()
        task = asyncio.create_task(self.predict())
        await until(lambda: self.started.is_set())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.app.state.gate.in_use, 1)
        self.assertTrue(Path(self.paths[0]).exists())
        self.block.set()
        await until(lambda: self.app.state.gate.in_use == 0)
        self.assertFalse(Path(self.paths[0]).exists())

    async def test_chunked_body_limit_closes_partial_upload(self):
        await self.reopen(max_body_bytes=180)
        body = (
            b"--BOUNDARY\r\nContent-Disposition: form-data; name=\"files\"; "
            b"filename=\"partial.png\"\r\nContent-Type: image/png\r\n\r\n"
            + b"x" * 400 + b"\r\n--BOUNDARY--\r\n"
        )

        async def chunks():
            yield body[:150]
            yield body[150:]

        response = await self.client.post(
            "/v1/predict", content=chunks(),
            headers={"Content-Type": "multipart/form-data; boundary=BOUNDARY"},
        )
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(self.app.state.gate.in_use, 0)
        self.assertEqual(list(Path(self.root.name).iterdir()), [])

    async def test_readiness_and_openapi(self):
        self.assertEqual((await self.client.get("/ready")).status_code, 200)
        schema = (await self.client.get("/openapi.json")).json()
        route = schema["paths"]["/v1/predict"]["post"]
        self.assertIn("multipart/form-data", route["requestBody"]["content"])
        self.app.state.scheduler.ready = False
        self.assertEqual((await self.client.get("/ready")).status_code, 503)
        self.assertEqual((await self.predict()).status_code, 500)

    async def test_upload_timeout_closes_partial_spool(self):
        await self.reopen(upload_timeout=0.03)
        spools = []
        original = tempfile.SpooledTemporaryFile

        def spool(*args, **kwargs):
            file = original(*args, **kwargs)
            spools.append(file)
            return file

        async def chunks():
            yield (
                b"--BOUNDARY\r\nContent-Disposition: form-data; name=\"files\"; "
                b"filename=\"partial.png\"\r\nContent-Type: image/png\r\n\r\npartial"
            )
            await asyncio.sleep(0.2)
            yield b"\r\n--BOUNDARY--\r\n"

        with patch("starlette.formparsers.SpooledTemporaryFile", side_effect=spool):
            response = await self.client.post(
                "/v1/predict", content=chunks(),
                headers={"Content-Type": "multipart/form-data; boundary=BOUNDARY"},
            )
        self.assertEqual(response.status_code, 408, response.text)
        self.assertTrue(spools)
        self.assertTrue(all(file.closed for file in spools))
        self.assertEqual(self.app.state.gate.in_use, 0)
        self.assertEqual(list(Path(self.root.name).iterdir()), [])

    async def test_default_batch_limit_boundary(self):
        response = await self.predict([image_bytes()] * 16)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [image["index"] for image in response.json()["results"]], list(range(16))
        )
        response = await self.predict([image_bytes()] * 17)
        self.assertEqual(response.status_code, 422, response.text)

    async def test_auth_rejects_before_admission_and_body_read_at_capacity(self):
        key = "test-key-012345678901234567890123456789"
        await self.reopen(api_key=key, require_api_key=True, max_requests=1)
        self.block.clear()
        first = asyncio.create_task(self.predict(headers={"X-API-Key": key}))
        await until(lambda: self.started.is_set())
        consumed = False

        async def stream():
            nonlocal consumed
            consumed = True
            yield b"invalid upload"

        for headers in ([], [("X-API-Key", "wrong")],
                        [("X-API-Key", key), ("X-API-Key", key)]):
            response = await asyncio.wait_for(self.client.post(
                "/v1/predict", content=stream(), headers=headers,
            ), 0.2)
            self.assertEqual(response.status_code, 401, response.text)
            self.assertFalse(consumed)
            self.assertEqual(len(self.app.state.gate.waiters), 0)
            self.assertEqual(self.app.state.gate.in_use, 1)
            self.assertNotIn(key, response.text)
        self.block.set()
        self.assertEqual((await first).status_code, 200)

    async def test_authenticated_swagger_and_public_health(self):
        key = "test-key-012345678901234567890123456789"
        await self.reopen(api_key=key, require_api_key=True)
        for path in ("/health", "/ready", "/docs", "/openapi.json"):
            self.assertEqual((await self.client.get(path)).status_code, 200)
        schema = (await self.client.get("/openapi.json")).json()
        self.assertEqual(schema["components"]["securitySchemes"]["DemoAPIKey"], {
            "type": "apiKey", "in": "header", "name": "X-API-Key",
        })
        self.assertEqual(schema["paths"]["/v1/predict"]["post"]["security"], [{"DemoAPIKey": []}])
        self.assertNotIn(key, str(schema))
        self.assertNotIn(key, repr(self.app.state.settings))
        self.assertEqual((await self.predict()).status_code, 401)
        self.assertEqual((await self.predict(headers={"X-API-Key": key})).status_code, 200)

    def test_hosted_settings_require_real_key_and_parse_boolean(self):
        for key in (None, "", "short", "REPLACE_WITH_RANDOM_KEY_AT_LEAST_32_CHARACTERS"):
            with self.assertRaises(ValueError):
                Settings(require_api_key=True, api_key=key)
        with patch.dict("os.environ", {"BOX_REQUIRE_API_KEY": "false"}, clear=True):
            self.assertFalse(Settings.from_env().require_api_key)
        with patch.dict("os.environ", {
            "BOX_REQUIRE_API_KEY": "true", "BOX_API_KEY": "a" * 32,
        }, clear=True):
            self.assertTrue(Settings.from_env().require_api_key)
        with patch.dict("os.environ", {"BOX_REQUIRE_API_KEY": "maybe"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
