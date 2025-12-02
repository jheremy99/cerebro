# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from rag import (
    extract_text_from_pdf_bytes,
    extract_text_from_docx_bytes,
    extract_text_from_txt_bytes,
    simple_chunk_text,
    embed_texts,
    build_faiss_index,
    save_index_and_meta,
    load_index_and_meta,
    retrieve_top_k,
    get_openai_client
)
import uuid
import numpy as np

app = FastAPI()

# CORS - permitir desde frontend (ajusta en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None
    index_name: Optional[str] = "default"  # cuál index usar

@app.post("/new-chat")
def new_chat():
    cid = str(uuid.uuid4())
    return {"chat_id": cid}

@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...), title: str = "uploaded", index_name: str = "default"):
    contents = await file.read()
    ext = file.filename.split(".")[-1].lower()
    if ext in ["pdf"]:
        text = extract_text_from_pdf_bytes(contents)
    elif ext in ["docx"]:
        text = extract_text_from_docx_bytes(contents)
    else:
        text = extract_text_from_txt_bytes(contents)

    # chunk
    chunks = simple_chunk_text(text, max_len=1000, overlap=200)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="No text extracted from file.")

    # embeddings
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embeddings error: {e}")

    index = build_faiss_index(embeddings)
    meta = {"chunks": chunks, "title": title}
    save_index_and_meta(index, meta, index_name)

    return {"status": "ok", "chunks": len(chunks), "index": index_name}

@app.post("/chat")
def chat(req: ChatRequest):
    # comprobar index
    index, meta = load_index_and_meta(req.index_name)
    if index is None:
        # fallback: no index yet
        return {"answer": "No hay documentos indexados. Suba un documento o use la IA base."}

    # embed pregunta
    client = get_openai_client()
    try:
        # crear embedding para pregunta (compatibilidad con la función embed_texts)
        q_emb = None
        if hasattr(client, "embeddings"):
            res = client.embeddings.create(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), input=req.question)
            q_emb = np.array(res.data[0].embedding, dtype="float32")
        else:
            import openai
            res = openai.Embedding.create(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), input=req.question)
            q_emb = np.array(res["data"][0]["embedding"], dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding question error: {e}")

    # recuperar top k
    ids, dists = retrieve_top_k(index, q_emb, k=4)
    retrieved_chunks = []
    for idx in ids:
        try:
            retrieved_chunks.append(meta["chunks"][int(idx)])
        except:
            pass

    # construir prompt simple
    context_text = "\n\n---\n\n".join(retrieved_chunks)
    system_prompt = "Eres un asistente académico que responde con claridad y paso a paso."
    user_prompt = f"{system_prompt}\nContexto:\n{context_text}\n\nPregunta: {req.question}\nRespuesta:"

    # llamar a OpenAI ChatCompletion
    try:
        # Usamos gpt-4o-mini vía la librería openai
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI chat error: {e}")

    return {"answer": answer, "retrieved": len(retrieved_chunks)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
