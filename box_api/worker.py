"""Process-local model ownership. No model is imported by the API process."""
import os
import warnings

from box_api.errors import InvalidImage

_model = None
_settings = None


def check_image(path, max_pixels):
    from PIL import Image, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                if image.format not in ("JPEG", "PNG", "BMP"):
                    raise InvalidImage("Supported formats are JPEG, PNG, and BMP")
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise InvalidImage(f"Image exceeds the {max_pixels} pixel limit")
                # Reads encoded data, but does not allocate a full decoded image.
                image.verify()
                return width, height
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Invalid or corrupt image") from exc


def initialize_worker(settings, barrier):
    global _model, _settings
    os.environ["OMP_NUM_THREADS"] = str(settings.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(settings.torch_threads)
    import cv2
    import numpy as np
    import torch
    from inference_core import load_model, predict_result

    torch.set_num_threads(settings.torch_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    _settings = settings
    _model = load_model(settings.model_path, settings.device)
    predict_result(np.zeros((640, 640, 3), dtype=np.uint8), _model, device=settings.device)
    # All processes must finish loading and warmup before readiness is enabled.
    barrier.wait(timeout=settings.startup_timeout)


def worker_ready():
    return os.getpid()


def infer_image(path, confidence):
    import cv2
    from inference_core import predict_result, serialize_predictions

    check_image(path, _settings.max_pixels)
    image = cv2.imread(path)
    if image is None:
        raise InvalidImage("Could not decode image")
    # OpenCV applies JPEG EXIF orientation; report the decoded dimensions.
    height, width = image.shape[:2]
    if width * height > _settings.max_pixels:
        raise InvalidImage("Decoded image exceeds the pixel limit")
    result = predict_result(image, _model, confidence, _settings.device)
    return {"width": width, "height": height, "predictions": serialize_predictions(result)}
