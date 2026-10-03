"""Opt-in test using the actual model and spawned CPU processes."""
import asyncio
import os
import unittest
from pathlib import Path

import httpx

from box_api.app import create_app
from box_api.settings import ROOT, Settings
from tests.test_api import image_bytes


@unittest.skipUnless(os.getenv("BOX_RUN_MODEL_TESTS") == "1", "Set BOX_RUN_MODEL_TESTS=1")
class ModelIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_spawned_pool_matches_sequential_model(self):
        image = image_bytes(64, 48)
        app = create_app(Settings(
            model_path=str(ROOT / "models" / "best.pt"),
            workers=2, processing_timeout=60,
            api_key="integration-test-key-01234567890123456789", require_api_key=True,
        ))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
                headers={"X-API-Key": app.state.settings.api_key},
            ) as client:
                self.assertEqual((await client.get("/ready")).status_code, 200)

                async def post():
                    return await client.post("/v1/predict", files=[
                        ("files", ("same.png", image, "image/png")),
                        ("files", ("same.png", image, "image/png")),
                    ])

                responses = await asyncio.gather(post(), post(), post())
                self.assertTrue(all(response.status_code == 200 for response in responses))
                for response in responses:
                    results = response.json()["results"]
                    self.assertEqual([result["index"] for result in results], [0, 1])
                    self.assertEqual([result["width"] for result in results], [64, 64])

                from inference_core import load_model, predict_result, serialize_predictions
                import cv2
                import numpy as np
                import torch
                torch.set_num_threads(1)
                model = await asyncio.to_thread(load_model, ROOT / "models" / "best.pt", "cpu")
                decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
                prediction = await asyncio.to_thread(predict_result, decoded, model, 0.5, "cpu")
                expected = serialize_predictions(prediction)
                for response in responses:
                    for result in response.json()["results"]:
                        self.assertEqual(result["predictions"], expected)


if __name__ == "__main__":
    unittest.main()
