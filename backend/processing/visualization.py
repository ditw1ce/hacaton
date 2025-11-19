import cv2
import numpy as np

def render_annotated_image(pil_image, faces):
    # faces: list of dicts {bbox:[x1,y1,x2,y2], label:"person_1"}
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for f in faces:
        x1, y1, x2, y2 = map(int, f["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f["label"], (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
