from facenet_pytorch import MTCNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def detect_faces(pil_image):
    boxes, probs = mtcnn.detect(pil_image)
    boxes = [] if boxes is None else boxes.tolist()
    probs = [] if probs is None else probs.tolist()
    return boxes, probs
