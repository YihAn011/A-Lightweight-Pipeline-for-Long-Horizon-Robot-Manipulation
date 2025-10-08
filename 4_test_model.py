#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

MODEL_PATH = "runs/detect/robotic_arm/weights/best.pt"
TEST_IMAGE_DIR = "traj/test1"
CONF_THRESHOLD = 0.022
OUTPUT_DIR = "test_results"

def test_model():
    print("="*70)
    print("Testing YOLO Model")
    print("="*70)
    
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found: {MODEL_PATH}")
        print("   Run 3_train_yolo.py first")
        return
    
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    test_images = sorted(Path(TEST_IMAGE_DIR).glob("*.jpg"))[:20]
    
    if len(test_images) == 0:
        print(f"Error: No images in: {TEST_IMAGE_DIR}")
        return
    
    print(f"Test images: {len(test_images)}\n")
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    for idx, img_path in enumerate(test_images):
        print(f"[{idx+1}/{len(test_images)}] Processing: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        
        results = model(image, conf=CONF_THRESHOLD, verbose=False)
        
        annotated = results[0].plot()
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                print(f"   Detected {i+1}: conf={conf:.2f}, Box=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
        else:
            print(f"   Warning: Nothing detected")
        
        output_path = Path(OUTPUT_DIR) / f"result_{idx:03d}.jpg"
        cv2.imwrite(str(output_path), annotated)
        
        cv2.imshow("Detection Result", annotated)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Testing completed!")
    print(f"Results in: {OUTPUT_DIR}/")
    print("="*70)

if __name__ == "__main__":
    test_model()
