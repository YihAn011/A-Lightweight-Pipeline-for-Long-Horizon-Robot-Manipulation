#!/usr/bin/env python3

from ultralytics import YOLO
import torch

DATA_YAML = "robotic_arm.yaml"
BASE_MODEL = "yolov8n.pt"

EPOCHS = 200
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    print("="*70)
    print("Starting YOLO Model Training")
    print("="*70)
    print(f"Configuration:")
    print(f"   Dataset: {DATA_YAML}")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   Training epochs: {EPOCHS}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Device: {DEVICE}")
    print("="*70)
    
    model = YOLO(BASE_MODEL)
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='runs/detect',
        name='robotic_arm',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        patience=100,
        save=True,
        save_period=20,
        cache=False,
        amp=False,
        
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.001,
        warmup_epochs=5,
        
        hsv_h=0.03,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.15,
        scale=0.7,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        
        close_mosaic=0,
        freeze=None,
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print(f"Model saved in: runs/detect/robotic_arm/weights/")
    print(f"   best.pt  - Best model")
    print(f"   last.pt  - Last epoch model")
    print("="*70)

if __name__ == "__main__":
    train_model()
