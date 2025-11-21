import sqlite3
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "face_db.sqlite")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def save_face(name, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb = np.array(embedding, dtype=np.float32)

    # ищем все эмбеддинги этого имени
    c.execute("SELECT embedding FROM faces WHERE name=?", (name,))
    rows = c.fetchall()

    if rows:
        # усредняем старые + новый
        all_embs = [np.frombuffer(r[0], dtype=np.float32) for r in rows]
        all_embs.append(emb)
        avg_emb = np.mean(all_embs, axis=0)
        emb_bytes = avg_emb.astype(np.float32).tobytes()
        # обновляем запись
        c.execute("UPDATE faces SET embedding=? WHERE name=?", (emb_bytes, name))
    else:
        # если имени нет — создаём новую запись
        emb_bytes = emb.astype(np.float32).tobytes()
        c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, emb_bytes))

    conn.commit()
    conn.close()

def find_matching_face(embedding, match_threshold=0.7):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM faces")
    rows = c.fetchall()
    conn.close()

    query_emb = np.array(embedding, dtype=np.float32)
    best_match = None
    best_sim = -1

    for name, emb_blob in rows:
        db_emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = cosine_similarity(query_emb, db_emb)
        if sim > best_sim:
            best_sim = sim
            best_match = name

    if best_sim >= match_threshold:
        return best_match
    return None

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a, b)
