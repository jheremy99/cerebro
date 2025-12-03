# main.py
import os
import uuid
import numpy as np
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    get_openai_client,
)

app = FastAPI(title="Academic AI Backend")

# CORS - revisar en producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    index_name: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

@app.post("/new-chat")
def new_chat():
    return {"conversation_id": str(uuid.uuid4())}

@app.post("/upload-doc")
async def upload_doc(
    file: UploadFile = File(...),
    title: str = "uploaded",
    index_name: str = "default",
):
    contents = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext == "pdf":
        text = extract_text_from_pdf_bytes(contents)
    elif ext == "docx":
        text = extract_text_from_docx_bytes(contents)
    else:
        text = extract_text_from_txt_bytes(contents)

    chunks = simple_chunk_text(text, max_len=int(os.getenv("CHUNK_SIZE", 1000)), overlap=int(os.getenv("CHUNK_OVERLAP", 200)))
    if not chunks:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto.")

    try:
        embeddings = embed_texts(chunks)
        index = build_faiss_index(embeddings)
        meta = {"chunks": chunks, "title": title}
        save_index_and_meta(index, meta, index_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error RAG: {e}")

    return {
        "success": True,
        "message": f"Documento {file.filename} indexado correctamente",
        "index": index_name,
        "chunks": len(chunks),
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or str(uuid.uuid4())

    index, meta = load_index_and_meta(req.index_name)
    retrieved_chunks = []

    if index is not None:
        try:
            client = get_openai_client()
            # embedding de la pregunta
            res = client.embeddings.create(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), input=req.message)
            q_emb = np.array(res.data[0].embedding, dtype="float32")

            ids, _ = retrieve_top_k(index, q_emb, k=4)
            for i in ids:
                try:
                    retrieved_chunks.append(meta["chunks"][int(i)])
                except Exception:
                    pass
        except Exception:
            # si falla recuperación/embeddings, seguimos sin contexto
            retrieved_chunks = []

    context_text = "\n\n---\n\n".join(retrieved_chunks)

    system_prompt = "Eres un asistente académico. Responde claro, paso a paso y solo con información confiable."

    user_message = f"""
Contexto:
{context_text}

Pregunta:
{req.message}
"""

    try:
        client = get_openai_client()
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=float(os.getenv("TEMPERATURE", 0.2)),
            max_tokens=int(os.getenv("MAX_TOKENS", 512)),
        )
        # obtener texto
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    return {"response": answer, "conversation_id": conversation_id}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
