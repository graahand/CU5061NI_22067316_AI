"""Artifiical Intelligence
    Bibek Poudel (22067316)"""

import cv2
import torch
from ultralytics import YOLO
import os
import json
from pathlib import Path

def run_inference():
    model_path = r'weights_yolo_seg\weights\best.pt'
    model = YOLO(model_path)
    OUTPUT_DIR = "shared_frames"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from webcam")
                break

            results = model.predict(
                task='segment',
                source=frame,
                conf=0.2
            )
            annotated_frame = results[0].plot()
            detections = []

            if hasattr(results[0], 'masks') and results[0].masks is not None:
                for seg in results[0].masks:
                    try:
                        conf = float(seg.conf[0].item()) if hasattr(seg, 'conf') and seg.conf is not None else 0.0
                        class_id = int(seg.cls[0].item()) if hasattr(seg, 'cls') and seg.cls is not None else -1
                        class_name = model.names[class_id] if class_id in model.names else "unknown"
                        area = len(seg.xy[0]) if hasattr(seg, 'xy') and seg.xy is not None else 0
                        detections.append({
                            "class_name": class_name,
                            "confidence_score": conf,
                            "area": area
                        })
                    except (AttributeError, IndexError):
                        continue
            
            cv2.imshow('YOLO Segmentation', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                frame_count += 1
                frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count}.jpg")
                metadata_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count}.json")
                cv2.imwrite(frame_path, annotated_frame)
                with open(metadata_path, "w") as meta_file:
                    json.dump({
                        "frame_number": frame_count,
                        "detections": detections,
                        "total_detections": len(detections)
                    }, meta_file, indent=4)
                print(f"Saved: {frame_path} with metadata: {metadata_path}")
                print(f"Number of detections: {len(detections)}")
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
