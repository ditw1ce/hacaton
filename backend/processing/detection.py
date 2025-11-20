from ultralytics import YOLO
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n-face-lindevs.pt")
yolo = YOLO(MODEL_PATH)

def detect_faces(pil_image, conf_thresh=0.5):
    results = yolo.predict(pil_image, verbose=False)
    boxes, probs = [], []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf.cpu().numpy())
            if conf < conf_thresh:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            boxes.append(xyxy)
            probs.append(conf)
            print('boxes', boxes)
            print('probs', probs)
    return boxes, probs
