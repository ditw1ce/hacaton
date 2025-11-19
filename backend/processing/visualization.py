import cv2
import numpy as np
from PIL import Image


def render_annotated_image(pil_image, faces):
    # Конвертируем PIL -> OpenCV BGR
    img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    for f in faces:
        x1, y1, x2, y2 = map(int, f["bbox"])
        # Рисуем рамку
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Рисуем текст метки, если есть
        label = f.get("label")
        if label is not None:
            cv2.putText(img, str(label), (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Конвертируем обратно в RGB для корректного сохранения через PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb
