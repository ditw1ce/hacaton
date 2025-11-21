import numpy as np
import cv2
import insightface

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))

def embed_face(pil_image, bbox):
    img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = model.get(img)

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        bx1, by1, bx2, by2 = bbox
        if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
            emb = f.embedding.astype(np.float32)
            norm = np.linalg.norm(emb) + 1e-8
            return (emb / norm).tolist()
    return None
