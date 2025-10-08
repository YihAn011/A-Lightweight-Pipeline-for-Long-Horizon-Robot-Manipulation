import cv2
import glob
from ultralytics import YOLO

frames = sorted(glob.glob("traj/test1/*.jpg"))
if not frames:
    print("Error: No images found")
    exit(1)

first_frame = cv2.imread(frames[0])
print(f"Image loaded: {frames[0]}")
print(f"   Size: {first_frame.shape}")

print("\nLoading YOLO model...")
yolo = YOLO("yolov8n.pt")

for conf_threshold in [0.25, 0.1, 0.05]:
    print(f"\n{'='*60}")
    print(f"Confidence threshold: {conf_threshold}")
    print('='*60)
    
    results = yolo(first_frame, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        print(f"Warning: No objects detected")
        continue
    
    print(f"Detected {len(boxes)} object(s):\n")
    
    class_names = results[0].names
    detected_items = []
    
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        class_name = class_names.get(cls_id, f"Unknown-{cls_id}")
        
        detected_items.append({
            'id': i+1,
            'class': class_name,
            'conf': conf,
            'box': (x1, y1, x2, y2)
        })
        
        print(f"  {i+1}. {class_name:20s} | Confidence: {conf:.3f} | Position: [{x1},{y1}] -> [{x2},{y2}]")
    
    vis_img = first_frame.copy()
    for item in detected_items:
        x1, y1, x2, y2 = item['box']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{item['class']} {item['conf']:.2f}"
        cv2.putText(vis_img, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_name = f"yolo_detection_conf{conf_threshold}.jpg"
    cv2.imwrite(output_name, vis_img)
    print(f"\nSaved: {output_name}")
    
    cv2.imshow(f"YOLO Detection (conf={conf_threshold})", vis_img)
    print("   Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


