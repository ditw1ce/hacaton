from PIL import Image
import os

def load_image(path):
    return Image.open(path).convert("RGB")

def list_images(dir_path, exts=(".jpg", ".jpeg", ".png")):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(exts)]
