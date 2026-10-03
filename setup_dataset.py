#!/usr/bin/env python3
"""
Dataset Setup Script for YOLO Box Condition Detection
دانلود و آماده‌سازی دیتاست از Roboflow

Usage:
    python setup_dataset.py --roboflow-key YOUR_API_KEY

    Or manually:
    1. Download from Roboflow
    2. Place in data/raw/
    3. Run: python setup_dataset.py --prepare-only
"""

import argparse
import logging
import shutil
import json
from pathlib import Path
from collections import defaultdict
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_roboflow_dataset(api_key, project_name="logistics", version=1):
    """Download dataset from Roboflow"""

    logger.info("Installing roboflow package...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'roboflow'])

    from roboflow import Roboflow

    logger.info(f"Downloading dataset: {project_name} v{version}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)
    dataset = project.download("yolov8", location="./data/raw")

    logger.info(f"✅ Dataset downloaded to: ./data/raw")
    return dataset


def prepare_yolo_dataset(raw_path="data/raw", processed_path="data/processed"):
    """
    Convert dataset to YOLO format (if needed)
    Assumes data is already in YOLO format from Roboflow
    """

    raw_path = Path(raw_path)
    processed_path = Path(processed_path)

    if not raw_path.exists():
        logger.error(f"❌ Raw dataset not found: {raw_path}")
        logger.info("Please download dataset from Roboflow first")
        return False

    logger.info(f"Preparing dataset from: {raw_path}")

    # Create processed directory structure
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (processed_path / split / subdir).mkdir(parents=True, exist_ok=True)

    # Copy images and labels
    src_images = raw_path / 'images'
    src_labels = raw_path / 'labels'

    if src_images.exists() and src_labels.exists():
        logger.info("Dataset is already in YOLO format")

        # List all images
        all_images = list(src_images.glob('**/*.jpg')) + list(src_images.glob('**/*.png'))
        logger.info(f"Found {len(all_images)} images")

        if len(all_images) == 0:
            logger.error("❌ No images found")
            return False

        # Split dataset (70% train, 20% val, 10% test)
        random.seed(42)
        random.shuffle(all_images)

        train_split = int(len(all_images) * 0.7)
        val_split = int(len(all_images) * 0.9)

        train_images = all_images[:train_split]
        val_images = all_images[train_split:val_split]
        test_images = all_images[val_split:]

        # Copy files
        for split_name, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            logger.info(f"Processing {split_name} set ({len(images)} images)...")

            for img_path in images:
                # Copy image
                dst_img = processed_path / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dst_img)

                # Copy label
                label_path = src_labels / img_path.stem + '.txt'
                if label_path.exists():
                    dst_label = processed_path / split_name / 'labels' / label_path.name
                    shutil.copy2(label_path, dst_label)

        logger.info(f"✅ Dataset split completed:")
        logger.info(f"   Train: {len(train_images)} images")
        logger.info(f"   Val:   {len(val_images)} images")
        logger.info(f"   Test:  {len(test_images)} images")

        return True

    else:
        logger.error("❌ Raw dataset structure not recognized")
        logger.info("Expected: data/raw/images/ and data/raw/labels/")
        return False


def validate_dataset(processed_path="data/processed"):
    """Validate dataset structure and content"""

    processed_path = Path(processed_path)

    logger.info("Validating dataset structure...")

    errors = []
    stats = {}

    for split in ['train', 'val', 'test']:
        images_dir = processed_path / split / 'images'
        labels_dir = processed_path / split / 'labels'

        if not images_dir.exists():
            errors.append(f"Missing: {images_dir}")
            continue

        images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        labels = list(labels_dir.glob('*.txt'))

        stats[split] = {
            'images': len(images),
            'labels': len(labels)
        }

        if len(images) == 0:
            errors.append(f"No images in {split}")

        if len(images) != len(labels):
            errors.append(f"Mismatch in {split}: {len(images)} images vs {len(labels)} labels")

    if errors:
        logger.warning("⚠️  Dataset validation warnings:")
        for error in errors:
            logger.warning(f"   {error}")
    else:
        logger.info("✅ Dataset structure is valid")

    logger.info("\nDataset statistics:")
    for split, counts in stats.items():
        logger.info(f"  {split:5s}: {counts['images']:4d} images, {counts['labels']:4d} labels")

    return len(errors) == 0


def create_symlink_if_needed():
    """Create symbolic link if raw data needs to be accessed"""

    raw_path = Path("data/raw")

    if not raw_path.exists():
        logger.info("To download dataset manually:")
        logger.info("1. Visit: https://universe.roboflow.com/large-benchmark-datasets/logistics-sz9jr")
        logger.info("2. Download YOLOv8 format dataset")
        logger.info("3. Extract to: data/raw/")
        logger.info("4. Run this script again")


def main():
    parser = argparse.ArgumentParser(
        description="Setup and prepare dataset for YOLO training"
    )

    parser.add_argument(
        '--roboflow-key',
        type=str,
        help='Roboflow API key for downloading dataset'
    )

    parser.add_argument(
        '--project',
        type=str,
        default='logistics',
        help='Roboflow project name'
    )

    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare existing dataset (skip download)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate dataset structure'
    )

    args = parser.parse_args()

    logger.info("="*50)
    logger.info("YOLO Dataset Setup")
    logger.info("="*50)

    # Download if API key provided
    if args.roboflow_key and not args.prepare_only:
        try:
            download_roboflow_dataset(args.roboflow_key, args.project)
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return

    elif not args.validate_only:
        create_symlink_if_needed()

    # Prepare dataset
    if not args.validate_only or (args.prepare_only or args.roboflow_key):
        if prepare_yolo_dataset():
            logger.info("✅ Dataset preparation completed")
        else:
            logger.error("❌ Dataset preparation failed")
            return

    # Validate
    if validate_dataset():
        logger.info("✅ Dataset is ready for training!")
        logger.info("\nNext steps:")
        logger.info("  1. Install dependencies: pip install -r requirements.txt")
        logger.info("  2. Start training: python train.py --epochs 50")
    else:
        logger.warning("⚠️  Dataset validation found issues")

    logger.info("="*50)


if __name__ == "__main__":
    main()
