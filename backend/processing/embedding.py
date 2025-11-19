import numpy as np
from PIL import Image
import insightface
import cv2

# Загружаем ArcFace
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(640, 640))


def crop_face(pil_image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return pil_image.crop((x1, y1, x2, y2))


def embed_face(pil_face):
    img = cv2.cvtColor(np.array(pil_face), cv2.COLOR_RGB2BGR)

    faces = model.get(img)

    if len(faces) == 0:
        return None

    emb = faces[0].embedding  # numpy array

    # L2 нормализация
    norm = np.linalg.norm(emb) + 1e-8
    return (emb / norm).tolist()
