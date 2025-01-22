"""Artifiical Intelligence
    Bibek Poudel (22067316)"""

import cv2
import os
import json
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
OUTPUT_DIR = "shared_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def object_detection_and_save_on_keypress(model, threshold=0.45):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame.")
            break

        results = model.predict(frame, conf=threshold)

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            detections.append({"class_name": class_name, "confidence_score": conf})

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            frame_count += 1
            frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count}.jpg")
            metadata_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count}.json")

            cv2.imwrite(frame_path, frame)

            with open(metadata_path, "w") as meta_file:
                json.dump(detections, meta_file)

            print(f"Saved: {frame_path} with metadata: {metadata_path}")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    object_detection_and_save_on_keypress(model)
