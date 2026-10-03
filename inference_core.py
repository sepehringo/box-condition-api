"""Shared inference and serialization; heavyweight libraries load only when needed."""
from pathlib import Path


def load_model(model_path, device="cpu"):
    from ultralytics import YOLO

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(str(model_path))
    model.to(device)
    return model


def predict_result(image, model, conf_threshold=0.5, device="cpu"):
    results = model.predict(
        source=image, conf=conf_threshold, device=device,
        imgsz=640, verbose=False, save=False,
    )
    return results[0] if results else None


def serialize_predictions(result):
    predictions = []
    if result is None or result.boxes is None:
        return predictions
    # Transfer arrays once instead of synchronizing the device for every box.
    boxes = result.boxes.cpu()
    for index, (class_id, confidence, coords) in enumerate(zip(
        boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
    )):
        class_id = int(class_id)
        predictions.append({
            "id": index,
            "class": result.names[class_id],
            "class_id": class_id,
            "confidence": round(float(confidence), 3),
            "box": dict(zip(
                ("x1", "y1", "x2", "y2"),
                (round(float(value), 2) for value in coords),
            )),
        })
    return predictions
