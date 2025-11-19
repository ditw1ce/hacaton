import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2').eval()

def crop_face(pil_image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return pil_image.crop((x1, y1, x2, y2))

def embed_face(pil_face):
    img = pil_face.resize((160, 160))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    import torch
    with torch.no_grad():
        tensor = torch.from_numpy(arr).unsqueeze(0)
        emb = model(tensor).numpy().squeeze()
    # L2-нормализация
    norm = np.linalg.norm(emb) + 1e-8
    return (emb / norm).tolist()
