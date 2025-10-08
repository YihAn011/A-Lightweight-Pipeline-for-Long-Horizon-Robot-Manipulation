#!/usr/bin/env python3

import cv2
import os
from pathlib import Path

DATASET_DIR = "yolo_dataset"
CLASS_NAMES = ["robotic_arm"]
CLASS_ID = 0

current_image = None
current_image_path = None
boxes = []
drawing = False
start_point = (0, 0)

def mouse_callback(event, x, y, flags, param):
    global current_image, boxes, drawing, start_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = current_image.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 255), 2)
        cv2.imshow("Annotation Tool", img_copy)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        x1, y1 = start_point
        x2, y2 = end_point
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        boxes.append((x1, y1, x2, y2))
        print(f"   Box {len(boxes)} added: [{x1},{y1},{x2},{y2}]")
        
        img_copy = current_image.copy()
        for (bx1, by1, bx2, by2) in boxes:
            cv2.rectangle(img_copy, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.imshow("Annotation Tool", img_copy)

def box_to_yolo_format(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def save_annotation(image_path, boxes_list):
    label_path = str(image_path).replace("/images/", "/labels/").replace(".jpg", ".txt")
    
    if len(boxes_list) == 0:
        with open(label_path, "w") as f:
            pass
        return
    
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    with open(label_path, "w") as f:
        for box in boxes_list:
            yolo_line = box_to_yolo_format(box, w, h)
            f.write(yolo_line + "\n")

def annotate_dataset():
    global current_image, current_image_path, boxes
    
    print("="*70)
    print("YOLO Image Annotation Tool")
    print("="*70)
    print("Keyboard shortcuts:")
    print("  - Mouse drag: Draw box")
    print("  - Enter: Save and next")
    print("  - D: Delete last box")
    print("  - S: Skip")
    print("  - Q: Quit")
    print("="*70)
    
    train_images = sorted(Path(DATASET_DIR).glob("images/train/*.jpg"))
    val_images = sorted(Path(DATASET_DIR).glob("images/val/*.jpg"))
    all_images = train_images + val_images
    
    if len(all_images) == 0:
        print("Error: No images found")
        return
    
    print(f"\nFound {len(all_images)} images")
    print(f"   Training: {len(train_images)}")
    print(f"   Validation: {len(val_images)}")
    print("\nStarting...\n")
    
    cv2.namedWindow("Annotation Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotation Tool", 1280, 960)
    cv2.setMouseCallback("Annotation Tool", mouse_callback)
    
    annotated_count = 0
    
    for idx, img_path in enumerate(all_images):
        current_image_path = img_path
        current_image = cv2.imread(str(img_path))
        boxes = []
        
        if current_image is None:
            print(f"Warning: Unable to read: {img_path}")
            continue
        
        print(f"\n[{idx+1}/{len(all_images)}] {img_path.name}")
        
        cv2.imshow("Annotation Tool", current_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:
                save_annotation(current_image_path, boxes)
                annotated_count += 1
                print(f"   Saved {len(boxes)} box(es)")
                break
            
            elif key == ord('d') or key == ord('D'):
                if len(boxes) > 0:
                    removed = boxes.pop()
                    print(f"   Deleted: {removed}")
                    img_copy = current_image.copy()
                    for (x1, y1, x2, y2) in boxes:
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow("Annotation Tool", img_copy)
            
            elif key == ord('s') or key == ord('S'):
                print(f"   Skipped")
                break
            
            elif key == ord('q') or key == ord('Q'):
                print(f"\n\nCompleted!")
                print(f"   Annotated: {annotated_count}/{len(all_images)}")
                cv2.destroyAllWindows()
                return
    
    cv2.destroyAllWindows()
    print("\n" + "="*70)
    print(f"All done!")
    print(f"   Annotated: {annotated_count}/{len(all_images)}")
    print("="*70)

if __name__ == "__main__":
    annotate_dataset()
