#!/usr/bin/env python3

import os
import shutil
import random
from pathlib import Path

SOURCE_FOLDERS = [
    "traj/test1",
    "traj/image_traj_20250831_213553",
    "traj/image_traj_20250831_213806",
    "traj/image_traj_20250831_215942",
    "traj/image_traj_20250901_114233",
]

OUTPUT_DIR = "yolo_dataset"
NUM_SAMPLES = 150
TRAIN_RATIO = 0.8

def prepare_dataset():
    print("="*70)
    print("Preparing YOLO Training Dataset")
    print("="*70)
    
    all_images = []
    for folder in SOURCE_FOLDERS:
        folder_path = Path(folder)
        if folder_path.exists():
            images = list(folder_path.glob("*.jpg"))
            all_images.extend(images)
            print(f"Found {len(images)} images from {folder}")
    
    print(f"\nTotal images found: {len(all_images)}")
    
    if len(all_images) < NUM_SAMPLES:
        print(f"Warning: Insufficient images, using all {len(all_images)} images")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, NUM_SAMPLES)
    
    train_img_dir = Path(OUTPUT_DIR) / "images" / "train"
    val_img_dir = Path(OUTPUT_DIR) / "images" / "val"
    train_label_dir = Path(OUTPUT_DIR) / "labels" / "train"
    val_label_dir = Path(OUTPUT_DIR) / "labels" / "val"
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    random.shuffle(selected_images)
    split_idx = int(len(selected_images) * TRAIN_RATIO)
    train_images = selected_images[:split_idx]
    val_images = selected_images[split_idx:]
    
    print(f"\nDataset split:")
    print(f"   Training set: {len(train_images)} images")
    print(f"   Validation set: {len(val_images)} images")
    
    print("\nCopying training images...")
    for idx, img in enumerate(train_images):
        new_name = f"train_{idx:04d}.jpg"
        shutil.copy(img, train_img_dir / new_name)
        label_file = train_label_dir / new_name.replace('.jpg', '.txt')
        label_file.write_text("")
        if (idx + 1) % 20 == 0:
            print(f"   Copied {idx + 1}/{len(train_images)} images")
    
    print("\nCopying validation images...")
    for idx, img in enumerate(val_images):
        new_name = f"val_{idx:04d}.jpg"
        shutil.copy(img, val_img_dir / new_name)
        label_file = val_label_dir / new_name.replace('.jpg', '.txt')
        label_file.write_text("")
        if (idx + 1) % 10 == 0:
            print(f"   Copied {idx + 1}/{len(val_images)} images")
    
    print(f"\nDataset prepared successfully! Saved in: {OUTPUT_DIR}/")
    print("="*70)

if __name__ == "__main__":
    prepare_dataset()
