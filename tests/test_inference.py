import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from box_api.errors import InvalidImage
from box_api.settings import Settings
from box_api.worker import check_image
from inference_core import predict_result, serialize_predictions


class InferenceTests(unittest.TestCase):
    def test_prediction_uses_requested_device_once(self):
        model = Mock()
        result = object()
        model.predict.return_value = [result]
        self.assertIs(predict_result("image", model, 0.7, "cpu"), result)
        model.predict.assert_called_once_with(
            source="image", conf=0.7, device="cpu",
            imgsz=640, verbose=False, save=False,
        )

    def test_box_serialization(self):
        result = Mock()
        boxes = result.boxes.cpu.return_value
        boxes.cls.tolist.return_value = [1]
        boxes.conf.tolist.return_value = [0.9876]
        boxes.xyxy.tolist.return_value = [[1.111, 2.222, 3.333, 4.444]]
        result.names = {1: "damaged"}
        self.assertEqual(serialize_predictions(result), [{
            "id": 0, "class": "damaged", "class_id": 1, "confidence": 0.988,
            "box": {"x1": 1.11, "y1": 2.22, "x2": 3.33, "y2": 4.44},
        }])
        self.assertEqual(serialize_predictions(None), [])

    def test_image_validation_supported_format_and_pixels(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "image")
            Image.new("RGB", (12, 10)).save(path, format="PNG")
            self.assertEqual(check_image(path, 120), (12, 10))
            with self.assertRaises(InvalidImage):
                check_image(path, 119)
            Image.new("RGB", (12, 10)).save(path, format="GIF")
            with self.assertRaises(InvalidImage):
                check_image(path, 1000)

    def test_environment_settings_validation(self):
        with patch.dict(os.environ, {"BOX_WORKERS": "4", "BOX_QUEUE_TIMEOUT": "1.5"}):
            settings = Settings.from_env()
        self.assertEqual(settings.workers, 4)
        self.assertEqual(settings.queue_timeout, 1.5)
        with self.assertRaises(ValueError):
            Settings(workers=0)
        with self.assertRaises(ValueError):
            Settings(device="mps", workers=2)

    def test_worker_accepts_exif_rotated_jpeg(self):
        from box_api import worker
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "rotated.jpg")
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (12, 20)).save(path, exif=exif)
            with patch.object(worker, "_settings", Settings()), \
                 patch.object(worker, "_model", object()), \
                 patch("inference_core.predict_result", return_value=None):
                result = worker.infer_image(path, 0.5)
        self.assertEqual((result["width"], result["height"]), (20, 12))
        self.assertEqual(result["predictions"], [])

    def test_cli_reuses_prediction_for_annotation_and_honors_cpu(self):
        import inference
        result = Mock()
        model = Mock()
        model._box_device = "cpu"
        with patch.object(inference.cv2, "imread", return_value="pixels"), \
             patch.object(inference, "predict_result", return_value=result) as predict, \
             patch.object(inference, "serialize_predictions", return_value=[]), \
             patch.object(inference, "draw_predictions") as draw:
            response = inference.predict_single_image(
                "test.png", model, annotate=True, output_path="out.jpg"
            )
        predict.assert_called_once_with("pixels", model, 0.5, "cpu")
        self.assertIs(draw.call_args.kwargs["result"], result)
        self.assertEqual(response["predictions"], [])


if __name__ == "__main__":
    unittest.main()
