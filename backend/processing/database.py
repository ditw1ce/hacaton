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
    emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
    # сохраняем эмбеддинг даже если имя уже есть
    c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, emb_bytes))
    conn.commit()
    conn.close()

def find_matching_face(embedding, threshold=0.9):
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

    if best_sim > (1 - threshold):
        return best_match
    return None

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a, b)
