import hashlib
import os
from typing import Dict, List, Tuple

import chromadb
import requests
import streamlit as st
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def read_uploaded_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raw = uploaded_file.read()
    return raw.decode("utf-8", errors="ignore")


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_documents(collection, embedder, docs: List[Tuple[str, str]]) -> int:
    all_chunks = []
    ids = []
    metadatas = []

    for source_name, text in docs:
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source_name}:{idx}:{chunk}".encode("utf-8")).hexdigest()
            ids.append(chunk_id)
            all_chunks.append(chunk)
            metadatas.append({"source": source_name, "chunk": idx})

    if not all_chunks:
        return 0

    existing_ids = set(collection.get(include=[], ids=ids)["ids"])
    new_payload = [(i, d, m) for i, d, m in zip(ids, all_chunks, metadatas) if i not in existing_ids]

    if not new_payload:
        return 0

    new_ids = [item[0] for item in new_payload]
    new_docs = [item[1] for item in new_payload]
    new_meta = [item[2] for item in new_payload]

    embeddings = embedder.encode(new_docs, normalize_embeddings=True).tolist()
    collection.add(ids=new_ids, documents=new_docs, metadatas=new_meta, embeddings=embeddings)
    return len(new_docs)


def retrieve_context(collection, embedder, query: str, top_k: int = 4):
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def make_openrouter_request(api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def build_messages(chat_history: List[Dict[str, str]], question: str, contexts: List[str]) -> List[Dict[str, str]]:
    context_text = "\n\n".join(contexts) if contexts else "(No retrieved context)"
    system_prompt = (
        "You are a document QA assistant. Use only the provided context and prior chat. "
        "If the answer is not present in context, say you don't know."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-8:])
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        }
    )
    return messages


def main() -> None:
    st.set_page_config(page_title="Simple Local RAG", layout="wide")
    st.title("Simple Local RAG Chat (Streamlit + OpenRouter + ChromaDB)")
    st.write("Upload local documents, index them, then chat with your docs.")

    embedder = get_embedder()
    collection = get_collection()
    default_api_key = os.getenv("OPENROUTER_API_KEY", "")
    default_model = os.getenv("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []

    with st.sidebar:
        st.subheader("LLM Settings")
        api_key = st.text_input("OpenRouter API Key", value=default_api_key, type="password")
        model_name = st.text_input("OpenRouter Model", value=default_model)
        top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=4, step=1)

        st.divider()
        st.subheader("Ingest Documents")
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, or .pdf files",
            accept_multiple_files=True,
            type=["txt", "md", "pdf"],
        )
        if st.button("Index Uploaded Files", type="primary"):
            if not uploaded_files:
                st.warning("Upload at least one file.")
            else:
                docs = []
                for file in uploaded_files:
                    text = read_uploaded_file(file)
                    docs.append((file.name, text))

                with st.spinner("Embedding and storing chunks..."):
                    inserted = add_documents(collection, embedder, docs)

                st.success(f"Indexed {inserted} new chunks.")

        if st.button("Reset Vector DB"):
            client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
            client.delete_collection(COLLECTION_NAME)
            client.get_or_create_collection(COLLECTION_NAME)
            st.success("Vector DB reset.")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_retrieved = []
            st.success("Chat cleared.")

    st.subheader("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask about your docs...")
    if question:
        if not api_key.strip():
            st.warning("Add your OpenRouter API key in the sidebar.")
            st.stop()

        count = collection.count()
        if count == 0:
            st.warning("No indexed data yet. Upload and index documents first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve_context(collection, embedder, question, top_k=top_k)

        contexts = [doc for doc, _ in retrieved if doc]
        st.session_state.last_retrieved = retrieved
        llm_messages = build_messages(st.session_state.messages[:-1], question, contexts)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    answer = make_openrouter_request(api_key, model_name, llm_messages)
                except requests.RequestException as exc:
                    st.error(f"OpenRouter request failed: {exc}")
                    st.stop()
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.last_retrieved:
        st.markdown("### Retrieved Context (Last Turn)")
        for idx, (doc, meta) in enumerate(st.session_state.last_retrieved, start=1):
            source = meta.get("source", "unknown") if meta else "unknown"
            chunk = meta.get("chunk", "?") if meta else "?"
            with st.expander(f"{idx}. {source} (chunk {chunk})"):
                st.write(doc)


if __name__ == "__main__":
    os.makedirs(DB_DIR, exist_ok=True)
    main()
