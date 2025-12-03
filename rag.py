# rag.py
import os
import pickle
import numpy as np
import faiss
from typing import List
from pypdf import PdfReader
import docx2txt
from openai import OpenAI

EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536  # ajustar si usas otro modelo

def extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(stream=data)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def extract_text_from_docx_bytes(data: bytes) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
        tf.write(data)
        temp_name = tf.name
    text = docx2txt.process(temp_name)
    try:
        os.remove(temp_name)
    except:
        pass
    return text

def extract_text_from_txt_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except:
        return data.decode("latin-1", errors="ignore")

def simple_chunk_text(text: str, max_len: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + max_len
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return [c.strip() for c in chunks if c.strip()]

def get_openai_client():
    """Return OpenAI v1 client (OpenAI class)."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: List[str]) -> np.ndarray:
    """Return stacked embeddings for a list of texts (float32 numpy array)."""
    client = get_openai_client()
    embeddings = []
    for t in texts:
        res = client.embeddings.create(model=EMBED_MODEL, input=t)
        emb = res.data[0].embedding
        embeddings.append(np.array(emb, dtype="float32"))
    return np.vstack(embeddings)

def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def save_index_and_meta(index: faiss.Index, meta: dict, filename: str):
    os.makedirs("indexes", exist_ok=True)
    faiss.write_index(index, f"indexes/{filename}.index")
    with open(f"indexes/{filename}.meta", "wb") as f:
        pickle.dump(meta, f)

def load_index_and_meta(filename: str):
    idx_path = f"indexes/{filename}.index"
    meta_path = f"indexes/{filename}.meta"
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        return None, None
    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def retrieve_top_k(index: faiss.Index, query_emb: np.ndarray, k: int = 4):
    D, I = index.search(np.array([query_emb]).astype("float32"), k)
    return I[0], D[0]
