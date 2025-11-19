from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov8n-face-lindevs.pt")

# Загружаем YOLO модель лиц
yolo = YOLO(MODEL_PATH)   # вес можно положить рядом в проекте

def detect_faces(pil_image):
    results = yolo.predict(pil_image, verbose=False)

    boxes = []
    probs = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()   # [x1,y1,x2,y2]
            conf = float(box.conf.cpu().numpy())

            boxes.append(xyxy)
            probs.append(conf)

    return boxes, probs
