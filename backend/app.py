# app.py
from utils.io import list_images, load_image
from utils.timer import Timer
from processing.embedding import embed_face   # убрали crop_face
from processing.clustering import cluster_embeddings, assign_person_ids
from processing.visualization import render_annotated_image
from processing.detection import detect_faces
import os, hashlib

def process_folder(folder, eps=0.5, min_samples=2):
    timer = Timer()
    image_paths = list_images(folder)
    all_faces, embeddings, index_map = [], [], []

    # Detect + Embed
    timer.start()
    for i, path in enumerate(image_paths):
        img = load_image(path)  # PIL.Image
        boxes, probs = detect_faces(img)
        print(f"{path}: YOLO нашёл {len(boxes)} боксов")
        faces_info = []
        for j, bbox in enumerate(boxes):
            emb = embed_face(img, bbox)   # передаём всё изображение и bbox
            if emb is None:
                faces_info.append({"bbox": bbox, "label": "no-embedding"})
                print(f"  лицо {j}: эмбеддинг не получен")
                continue
            embeddings.append(emb)
            index_map.append((i, j))
            faces_info.append({"bbox": bbox, "label": None})
        all_faces.append({"path": path, "image": img, "faces": faces_info})
        print(f"{path}: добавлено {len(faces_info)} лиц")
    timer.stop("Detect+Embed")

    # Cluster
    timer.start()
    if embeddings:
        labels = cluster_embeddings(embeddings, eps=eps, min_samples=min_samples, metric='cosine')
        person_ids = assign_person_ids(labels)
        for k, (img_idx, face_idx) in enumerate(index_map):
            all_faces[img_idx]["faces"][face_idx]["label"] = person_ids[k]
    else:
        print("⚠️ Нет лиц для кластеризации")
    timer.stop("Cluster")

    # Render and save
    os.makedirs("results", exist_ok=True)
    for item in all_faces:
        annotated_pil = render_annotated_image(item["image"], item["faces"])  # возвращает PIL.Image
        name, ext = os.path.splitext(os.path.basename(item["path"]))
        hash_suffix = hashlib.md5(item["path"].encode()).hexdigest()[:6]
        out_path = os.path.join("results", f"{name}_{hash_suffix}{ext}")
        annotated_pil.save(out_path)
        print(f"Сохранено: {out_path}")

    return all_faces

if __name__ == "__main__":
    folder = "data_examples"
    results = process_folder(folder, eps=0.5, min_samples=2)
    print(f"Processed {len(results)} images. See results/ for annotated outputs.")
