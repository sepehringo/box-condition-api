#!/usr/bin/env python3
"""
YOLOv8 Inference Script for Box Condition Detection
پیش‌بینی سلامت جعبه (صدمه‌دیده/سالم)

Usage:
    # Single image
    python inference.py --image path/to/image.jpg

    # Multiple images from folder
    python inference.py --image data/processed/test/images

    # Webcam
    python inference.py --source 0

    # Custom model
    python inference.py --image test.jpg --model models/custom.pt
"""

import argparse
import logging
from pathlib import Path
from inference_core import load_model as core_load_model, predict_result, serialize_predictions
import cv2
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path="models/best.pt", device="mps"):
    """Load trained YOLO model"""
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Please train a model first using train.py")
        return None

    logger.info(f"Loading model: {model_path}")
    model = core_load_model(model_path, device)
    model._box_device = device
    logger.info(f"✅ Model loaded successfully on device: {device}")

    return model


def predict_single_image(image_path, model, conf_threshold=0.5, device=None,
                         annotate=False, output_path=None):
    """Predict once, optionally using the same result for annotations."""
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return None
    device = device or getattr(model, "_box_device", "cpu")
    result = predict_result(img, model, conf_threshold, device)
    if annotate and result is not None:
        draw_predictions(image_path, model, conf_threshold, output_path,
                         device=device, result=result)
    return {
        "image": str(image_path),
        "predictions": serialize_predictions(result),
        "timestamp": datetime.now().isoformat(),
    }


def draw_predictions(image_path, model, conf_threshold=0.5, output_path=None,
                     device=None, result=None):
    """Draw a cached prediction, or predict if called independently."""
    if result is None:
        img = cv2.imread(image_path)
        if img is None:
            return None
        device = device or getattr(model, "_box_device", "cpu")
        result = predict_result(img, model, conf_threshold, device)
    if result is None:
        return None
    if output_path is None:
        output_dir = Path("results/inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_path).stem}_detected.jpg"
    if not cv2.imwrite(str(output_path), result.plot()):
        raise OSError(f"Could not save annotated image: {output_path}")
    logger.info(f"Saved: {output_path}")
    return str(output_path)

def predict_folder(folder_path, model, conf_threshold=0.5, output_json=True, device=None):
    """Run inference on all images in a folder"""

    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"❌ Folder not found: {folder_path}")
        return None

    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in folder.rglob('*')
              if f.suffix.lower() in image_extensions]

    logger.info(f"Found {len(images)} images in {folder_path}")

    all_results = []
    for idx, image_path in enumerate(images, 1):
        logger.info(f"[{idx}/{len(images)}] Processing {image_path.name}")

        result = predict_single_image(str(image_path), model, conf_threshold,
                                      device=device, annotate=True)
        if result:
            all_results.append(result)

    # Save JSON results
    if output_json:
        output_dir = Path("results/inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"✅ Results saved to: {json_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with YOLOv8 box condition detection model"
    )

    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file or folder with images'
    )

    parser.add_argument(
        '--source',
        type=int,
        default=None,
        help='Webcam source (0 = default camera)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/best.pt',
        help='Path to trained model'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold (0-1)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for annotated image/results'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Save results as JSON'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cpu'],
        help='Inference device'
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model, args.device)
    if model is None:
        return

    # Run inference
    if args.image:
        image_path = Path(args.image)

        if image_path.is_dir():
            # Folder of images
            logger.info(f"Processing folder: {args.image}")
            predict_folder(args.image, model, args.conf, args.json, device=args.device)
        else:
            # Single image
            logger.info(f"Processing single image: {args.image}")
            result = predict_single_image(args.image, model, args.conf,
                                          device=args.device, annotate=True,
                                          output_path=args.output)

            if result:
                print("\n" + "="*50)
                print("PREDICTIONS:")
                print("="*50)
                print(json.dumps(result, indent=2))
                print("="*50 + "\n")

    elif args.source is not None:
        # Webcam inference
        logger.info(f"Starting webcam inference (camera {args.source})")
        cap = cv2.VideoCapture(args.source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = model.predict(
                source=frame,
                conf=args.conf,
                device=args.device,
                verbose=False
            )

            if results:
                annotated_frame = results[0].plot()
                cv2.imshow("Box Condition Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        parser.print_help()
        logger.error("Please provide --image or --source")


if __name__ == "__main__":
    main()
