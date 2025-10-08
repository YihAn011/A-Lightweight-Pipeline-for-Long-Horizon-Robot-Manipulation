import sys
import os
import cv2
import numpy as np
import glob
import torch
import random


project_root = os.path.dirname(os.path.abspath(__file__))
xmem_root = os.path.join(project_root, "XMem")


if xmem_root not in sys.path:
    sys.path.append(xmem_root)

print("sys.path loaded:")
for p in sys.path:
    if "XMem" in p:
        print("   ", p)



from inference.inference_core import InferenceCore
from model.network import XMem
from util.tensor_util import pad_divide_by


from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XMEM_WEIGHTS = os.path.join(xmem_root, "saves", "XMem.pth")

YOLO_MODEL = "yolov8n.pt"


def get_initial_mask_manual(image, sam_model_path):
    import gc
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_model_path)
    predictor = SamPredictor(sam)
    
    rectangles = []
    drawing = False
    start_pt = (0, 0)
    window_name = "Manual Selection"
    
    def mouse_draw(event, x, y, flags, param):
        nonlocal drawing, start_pt
        img_copy = image.copy()
        
        for (p1, p2) in rectangles:
            cv2.rectangle(img_copy, p1, p2, (0, 255, 0), 3)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.rectangle(img_copy, start_pt, (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_pt = (x, y)
            rectangles.append((start_pt, end_pt))
            cv2.rectangle(img_copy, start_pt, end_pt, (0, 255, 0), 3)
            cv2.imshow(window_name, img_copy)
            print(f"   Box {len(rectangles)} selected: {start_pt} -> {end_pt}")
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, mouse_draw)
    
    print("\n" + "="*60)
    print("Manual Selection Mode")
    print("="*60)
    print("="*60)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        elif key == 32:
            continue
    
    cv2.destroyAllWindows()
    
    if len(rectangles) == 0:
        print("Warning: No region selected, returning empty mask")
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    
    for (p1, p2) in rectangles:
        cx, cy = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]
        combined_mask |= best_mask
    
    del sam, predictor
    gc.collect()
    print("Manual segmentation completed")
    
    return combined_mask.astype(np.uint8)


def get_initial_mask_with_yolo_sam(image, sam_model_path, yolo_model_path=YOLO_MODEL, 
                                    target_classes=None, conf_threshold=0.015):
    import gc
    
    yolo = YOLO(yolo_model_path)
    
    results = yolo(image, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes
    
    print(f"\nYOLO Detection Results: {len(boxes)} object(s) detected")
    if len(boxes) > 0:
        class_names = results[0].names
        detected_classes = {}
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = class_names.get(cls_id, f"Unknown-{cls_id}")
            if class_name not in detected_classes:
                detected_classes[class_name] = []
            detected_classes[class_name].append((i, conf))
        
        for class_name, items in detected_classes.items():
            print(f"  - {class_name}: {len(items)} object(s) (confidence: {[f'{c:.2f}' for _, c in items]})")
    else:
        print("  Warning: No objects detected")
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    print(f"\nYOLO detected {len(boxes)} object(s)")
    
    vis_img = image.copy()
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cls_id = int(boxes.cls[i].item())
        conf = boxes.conf[i].item()
        
        if target_classes is not None and cls_id not in target_classes:
            continue
        
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {cls_id}: {conf:.2f}"
        cv2.putText(vis_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLO Detection", vis_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    del yolo
    gc.collect()
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_model_path)
    predictor = SamPredictor(sam)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cls_id = int(boxes.cls[i].item())
        
        if target_classes is not None and cls_id not in target_classes:
            continue
        
        print(f"  Segmenting object {i+1}: Box [{x1},{y1},{x2},{y2}]")
        
        masks, scores, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=True
        )
        
        best_mask = masks[np.argmax(scores)]
        combined_mask |= best_mask
    
    print(f"SAM segmentation completed, object pixels in mask: {combined_mask.sum()}")
    
    mask_vis = (combined_mask * 255).astype(np.uint8)
    colored_mask = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
    cv2.imshow("Final Mask", np.hstack((image, overlay)))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    del sam, predictor
    gc.collect()
    print("YOLO + SAM models released")
    
    return combined_mask.astype(np.uint8)


def process_video_with_xmem(input_folder, model_path_sam, output_path, 
                             use_yolo=True, yolo_model=YOLO_MODEL, target_classes=None):
    frames = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
    if not frames:
        raise FileNotFoundError(f"No images found in: {input_folder}")
    print(f"Found {len(frames)} images")

    first_frame = cv2.imread(frames[0])
    
    if use_yolo:
        print("Attempting to use YOLO + SAM automatic detection mode")
        init_mask = get_initial_mask_with_yolo_sam(
            first_frame, model_path_sam, 
            yolo_model_path=yolo_model,
            target_classes=target_classes,
            conf_threshold=0.015
        )
        
        if init_mask.sum() == 0:
            print("\n" + "="*70)
            print("YOLO still did not detect any objects")
            print("="*70)
            print("="*70)
            return
    else:
        print("Using manual selection mode")
        init_mask = get_initial_mask_manual(first_frame, model_path_sam)
    
    cv2.imwrite("init_mask.png", init_mask * 255)
    print(f"Initial mask generated, pixels: {init_mask.sum()}")
    
    import time
    time.sleep(2)
    print("Preparing to load XMem model...")

    config = {
        'top_k': 20,
        'mem_every': 10,
        'deep_update_every': -1,
        'enable_long_term': False,
        'enable_long_term_count_usage': False,
        'num_prototypes': 128,
        'min_mid_term_frames': 3,
        'max_mid_term_frames': 5,
        'max_long_term_elements': 10000,
    }
    print("Using low memory configuration")
    
    network = XMem(config, XMEM_WEIGHTS, map_location=DEVICE).to(DEVICE).eval()
    
    processor = InferenceCore(network, config=config)
    processor.set_all_labels([1])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = first_frame.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

    rgb_torch = torch.from_numpy(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    rgb_torch = rgb_torch.to(DEVICE)
    
    mask_torch = torch.from_numpy(init_mask).unsqueeze(0).float().to(DEVICE)
    
    prob = processor.step(rgb_torch, mask_torch, valid_labels=[1])
    
    out_mask = torch.argmax(prob, dim=0)
    mask_first = (out_mask > 0).cpu().numpy().astype(np.uint8)
    
    cv2.imwrite("xmem_first_frame_mask.png", mask_first * 255)
    print(f"Debug info: prob shape={prob.shape}, object pixels in mask={mask_first.sum()}")
    
    blurred = cv2.GaussianBlur(first_frame, (51, 51), 0)
    result = np.where(mask_first[..., None], first_frame, blurred)
    out.write(result)
    
    cv2.imshow("First Frame Result", np.hstack((first_frame, result)))
    cv2.waitKey(1000)


    import gc
    
    for idx, path in enumerate(frames[1:], start=1):
        frame = cv2.imread(path)
        
        if frame is None:
            print(f"Warning: Frame {idx+1} read failed, skipping: {path}")
            continue
        
        rgb_torch = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        rgb_torch = rgb_torch.to(DEVICE)
        
        prob = processor.step(rgb_torch)
        
        out_mask = torch.argmax(prob, dim=0)
        mask = (out_mask > 0).cpu().numpy().astype(np.uint8)
        
        blurred = cv2.GaussianBlur(frame, (51, 51), 0)
        result = np.where(mask[..., None], frame, blurred)
        out.write(result)
        
        cv2.imshow("Result", np.hstack((frame, result)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print(f"Processed frame {idx+1}/{len(frames)}")
        
        if idx % 20 == 0:
            gc.collect()
            print(f"Memory cleaned (frame {idx+1})")


    out.release()
    cv2.destroyAllWindows()
    print(f"Video generated: {output_path}")


def main():
    input_folder = "traj/test1"
    model_path_sam = "sam_vit_h_4b8939.pth"
    output_path = "/Users/yihe/Desktop/2025_Fall_Columbia/robotics/project/output_xmem_yolo.mp4"
    
    use_yolo = True
    yolo_model = "runs/detect/robotic_arm/weights/best.pt"
    
    target_classes = None
    
    process_video_with_xmem(
        input_folder, 
        model_path_sam, 
        output_path,
        use_yolo=use_yolo,
        yolo_model=yolo_model,
        target_classes=target_classes
    )

if __name__ == "__main__":
    main()
