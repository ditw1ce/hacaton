from utils.io import list_images, load_image
from utils.timer import Timer
from processing.detection import detect_faces
from processing.embedding import crop_face, embed_face
from processing.clustering import cluster_embeddings, assign_person_ids
from processing.visualization import render_annotated_image
from PIL import Image
import os

def process_folder(folder, eps=0.5, min_samples=2):
    timer = Timer()
    image_paths = list_images(folder)
    all_faces = []   # per-image faces info
    embeddings = []  # flat list of embeddings
    index_map = []   # (image_idx, face_idx) for each embedding

    # Detect + Embed
    timer.start()
    for i, path in enumerate(image_paths):
        img = load_image(path)
        boxes, probs = detect_faces(img)
        faces_info = []
        if boxes:
            for j, bbox in enumerate(boxes):
                face_img = crop_face(img, bbox)
                emb = embed_face(face_img)
                embeddings.append(emb)
                index_map.append((i, j))
                faces_info.append({"bbox": bbox, "label": None})
        all_faces.append({"path": path, "image": img, "faces": faces_info})
    timer.stop("Detect+Embed")

    # Cluster
    timer.start()
    labels = cluster_embeddings(embeddings, eps=eps, min_samples=min_samples, metric='cosine')
    person_ids = assign_person_ids(labels)
    timer.stop("Cluster")

    # Assign labels back
    for k, (img_idx, face_idx) in enumerate(index_map):
        all_faces[img_idx]["faces"][face_idx]["label"] = person_ids[k]

    # Render and save
    os.makedirs("results", exist_ok=True)
    for item in all_faces:
        annotated = render_annotated_image(item["image"], item["faces"])
        out_path = os.path.join("results", os.path.basename(item["path"]))
        Image.fromarray(annotated).save(out_path)
    return all_faces

if __name__ == "__main__":
    folder = "data_examples"
    results = process_folder(folder, eps=0.5, min_samples=2)
    print(f"Processed {len(results)} images. See results/ for annotated outputs.")
