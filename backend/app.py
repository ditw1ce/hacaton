from utils.io import list_images, load_image
from utils.timer import Timer
from processing.embedding import embed_face 
from processing.visualization import render_annotated_image
from processing.detection import detect_faces
from processing.database import init_db, find_matching_face, save_face
from processing.clustering import cluster_embeddings, assign_person_ids
import os, hashlib

def process_folder(folder, mode="interactive", eps=0.5, min_samples=2):
    """
    mode = "interactive" -> спрашивает имена для новых кластеров
    mode = "auto"        -> автоматически присваивает person_1, person_2...
    """
    init_db()
    timer = Timer()
    image_paths = list_images(folder)
    all_faces, embeddings, index_map = [], [], []
    unknown_embeddings, unknown_index_map = [], []

    # Detect + Embed
    timer.start()
    for i, path in enumerate(image_paths):
        img = load_image(path)
        boxes, probs = detect_faces(img)
        print(f"{path}: YOLO нашёл {len(boxes)} боксов")
        faces_info = []

        for j, bbox in enumerate(boxes):
            emb = embed_face(img, bbox)
            if emb is None:
                faces_info.append({"bbox": bbox, "label": "no-embedding"})
                print(f"  лицо {j}: эмбеддинг не получен")
                continue

            name = find_matching_face(emb)
            if name:
                label = name
                print(f"  лицо {j}: найдено в базе как {label}")
                embeddings.append(emb)
                index_map.append((i, j))
            else:
                label = "unknown"
                unknown_embeddings.append(emb)
                unknown_index_map.append((i, j))
                print(f"  лицо {j}: неизвестно, добавлено в кластеризацию")

            faces_info.append({"bbox": bbox, "label": label})

        all_faces.append({"path": path, "image": img, "faces": faces_info})
        print(f"{path}: добавлено {len(faces_info)} лиц")
    timer.stop("Detect+Embed")

    # Cluster только для unknown
    timer.start()
    if unknown_embeddings:
        labels = cluster_embeddings(unknown_embeddings, eps=eps, min_samples=min_samples, metric='cosine')
        person_ids = assign_person_ids(labels)

        cluster_name_map = {}
        for lbl in set(person_ids):
            if mode == "auto":
                # просто присваиваем person_X
                cluster_name_map[lbl] = lbl if lbl != "unknown" else "unknown"
            else:
                # interactive: показываем превью и спрашиваем имя
                if lbl == "unknown":
                    continue
                idx = person_ids.index(lbl)
                img_idx, face_idx = unknown_index_map[idx]
                face_bbox = all_faces[img_idx]["faces"][face_idx]["bbox"]
                pil_img = all_faces[img_idx]["image"]

                x1, y1, x2, y2 = map(int, face_bbox)
                face_crop = pil_img.crop((x1, y1, x2, y2))
                preview_path = os.path.join("results", f"preview_{lbl}.jpg")
                face_crop.save(preview_path)
                print(f"Открой {preview_path}, чтобы увидеть лицо для {lbl}")

                name = input(f"Введите имя для {lbl}: ").strip()
                cluster_name_map[lbl] = name if name else "unknown"

        # присваиваем имена и сохраняем в базу
        for k, (img_idx, face_idx) in enumerate(unknown_index_map):
            cluster_label = person_ids[k]
            final_name = cluster_name_map.get(cluster_label, cluster_label)
            all_faces[img_idx]["faces"][face_idx]["label"] = final_name
            if final_name != "unknown" and mode == "interactive":
                save_face(final_name, unknown_embeddings[k])
                print(f"  лицо {face_idx}: сохранено как {final_name}")
    else:
        print("⚠️ Нет новых лиц для кластеризации")
    timer.stop("Cluster")

    # Render and save
    os.makedirs("results", exist_ok=True)
    for item in all_faces:
        annotated_pil = render_annotated_image(item["image"], item["faces"])
        name, ext = os.path.splitext(os.path.basename(item["path"]))
        hash_suffix = hashlib.md5(item["path"].encode()).hexdigest()[:6]
        out_path = os.path.join("results", f"{name}_{hash_suffix}{ext}")
        annotated_pil.save(out_path)
        print(f"Сохранено: {out_path}")

    return all_faces

if __name__ == "__main__":
    folder = "data_examples"
    # выбери режим: "auto" или "interactive"
    results = process_folder(folder, mode="auto")
    print(f"Processed {len(results)} images. See results/ for annotated outputs.")
