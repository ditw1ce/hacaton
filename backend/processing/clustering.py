import numpy as np
from sklearn.cluster import DBSCAN

def cluster_embeddings(embeddings, eps=0.3, min_samples=2, metric='cosine'):
    if len(embeddings) == 0:
        return []
    X = np.array(embeddings, dtype=np.float32)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(X)
    return labels.tolist()

def assign_person_ids(labels):
    id_map = {}
    next_id = 1
    person_ids = []
    for lbl in labels:
        if lbl == -1:
            person_ids.append("unknown")
        else:
            if lbl not in id_map:
                id_map[lbl] = f"person_{next_id}"
                next_id += 1
            person_ids.append(id_map[lbl])
    return person_ids
