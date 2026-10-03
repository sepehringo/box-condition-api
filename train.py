#!/usr/bin/env python3
"""
YOLOv8 Training Script for Box Condition Detection
تشخیص سلامت جعبه (صدمه‌دیده/سالم)

Usage:
    python train.py --epochs 50 --batch 16
    python train.py --device mps --img 640
"""

import argparse
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO
import torch

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device():
    """Setup training device (MPS for M5 Pro)"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "mps":
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Metal GPU available: True")

    return device


def train_model(args):
    """Main training function"""

    # Setup
    logger.info("=" * 50)
    logger.info("Box Condition Detection - YOLOv8 Training")
    logger.info("Dataset: Package Damage Recognition (AIDL) - 355 images")
    logger.info("Classes: damaged, intact")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Config loaded from: {args.config}")

    # Setup device
    device = setup_device()

    # Override config with CLI arguments
    model_name = args.model or config['model']['name']
    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['training']['batch_size']
    img_size = args.img_size or 640

    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Image Size: {img_size}x{img_size}")

    # Load or create model
    logger.info(f"Loading YOLOv8{model_name[-1]} model...")
    model = YOLO(f"yolov8{model_name[-1]}.pt")

    # Verify data.yaml exists
    data_yaml_path = Path("custom_data.yaml")
    if not data_yaml_path.exists():
        logger.error(f"❌ custom_data.yaml not found at {data_yaml_path}")
        logger.error("Please ensure custom_data.yaml exists in the project root")
        return None

    logger.info(f"Dataset config: {data_yaml_path}")

    # Train the model
    logger.info("Starting training...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=config['training']['patience'],
        device=device,
        project="results",
        name="yolo_box_detection",
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=5,
        val=True,
        conf=config.get('validation', {}).get('conf_threshold', 0.5),
        iou=config.get('validation', {}).get('iou_threshold', 0.45),
        # Data augmentation
        hsv_h=config['augmentation']['hsv_h'],
        hsv_s=config['augmentation']['hsv_s'],
        hsv_v=config['augmentation']['hsv_v'],
        degrees=config['augmentation']['degrees'],
        translate=config['augmentation']['translate'],
        scale=config['augmentation']['scale'],
        flipud=config['augmentation']['flipud'],
        fliplr=config['augmentation']['fliplr'],
        mosaic=config['augmentation']['mosaic'],
    )

    logger.info("✅ Training completed!")

    # Save best model to models directory
    best_model_path = Path("results/yolo_box_detection/weights/best.pt")
    if best_model_path.exists():
        import shutil
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        final_path = models_dir / "best.pt"
        shutil.copy(str(best_model_path), str(final_path))
        logger.info(f"✅ Best model saved to: {final_path}")

    return model, results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for box condition detection"
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=['yolov8n', 'yolov8s', 'yolov8m'],
        help='Model size (nano/small/medium). Default: yolov8n'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        dest='batch_size',
        help='Batch size (default: 16)'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        dest='img_size',
        help='Input image size (default: 640)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cpu'],
        help='Training device'
    )

    args = parser.parse_args()

    # Train
    model, results = train_model(args)

    logger.info("=" * 50)
    logger.info("Training script finished!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
