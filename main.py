import os
import tempfile
from typing import List, Tuple

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.documents import Document


def load_pdfs_to_docs(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        docs.extend(PyPDFLoader(tmp_path).load())
    return docs


def build_vectorstore(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    return FAISS.from_documents(chunks, embeddings)


def format_sources(source_docs: List[Document]) -> str:
    if not source_docs:
        return ""
    lines = []
    for i, d in enumerate(source_docs[:6], start=1):
        meta = d.metadata or {}
        src = os.path.basename(str(meta.get("source", "PDF")))
        page = meta.get("page")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
        if page is not None:
            lines.append(f"{i}. {src} — page {page + 1}: {snippet}")
        else:
            lines.append(f"{i}. {src}: {snippet}")
    return "\n".join(lines)


def build_context(docs: List[Document], max_chars: int = 12000) -> str:
    """Join retrieved chunks into a context block, capped to avoid huge prompts."""
    parts = []
    total = 0
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 Gemini RAG Chatbot")

if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "Missing GOOGLE_API_KEY.\n\n"
        "Set it in your terminal and re-run:\n"
        "export GOOGLE_API_KEY='YOUR_KEY'"
    )
    st.stop()

with st.sidebar:
    st.header("Upload & Index")

    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    st.subheader("Gemini settings")
    chat_model = st.text_input("Chat model", value="gemini-2.0-flash")
    embedding_model = st.text_input("Embedding model", value="gemini-embedding-001")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.subheader("RAG settings")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=150, step=50)
    top_k = st.number_input("Top K (retrieval)", min_value=1, max_value=20, value=4, step=1)

    build_btn = st.button("Build / Rebuild Index", type="primary")


# ---------------------------
# Session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Tuple[str, str]] = []


# ---------------------------
# Build index
# ---------------------------
if build_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        try:
            with st.spinner("Loading PDFs..."):
                docs = load_pdfs_to_docs(uploaded_files)

            with st.spinner("Building vector index..."):
                vs = build_vectorstore(
                    docs,
                    int(chunk_size),
                    int(chunk_overlap),
                    embedding_model,
                )
                st.session_state.vectorstore = vs

            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("Index built! Start chatting.")
        except Exception as e:
            st.session_state.vectorstore = None
            st.error(f"Failed to build index: {e}")


# Render chat so far
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ---------------------------
# Chat
# ---------------------------
prompt = st.chat_input("Ask a question about your PDF(s)...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.error("Upload PDF(s) and click **Build / Rebuild Index** first.")
    else:
        try:
            # Retrieve relevant chunks
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": int(top_k)})
            retrieved_docs = retriever.invoke(prompt)

            context = build_context(retrieved_docs)

            # Keep a small conversation history in the prompt
            history_text = ""
            for u, a in st.session_state.chat_history[-6:]:
                history_text += f"User: {u}\nAssistant: {a}\n"

            system_prompt = (
                "You are a helpful assistant. Use ONLY the provided context to answer.\n"
                "If the answer is not in the context, say: 'I don't know based on the uploaded documents.'\n"
                "Cite pages when possible."
            )

            full_prompt = f"""{system_prompt}

Conversation so far:
{history_text}

Context:
{context}

User question:
{prompt}
"""

            llm = ChatGoogleGenerativeAI(model=chat_model, temperature=float(temperature))

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = llm.invoke(full_prompt)

                answer = getattr(response, "content", str(response)).strip()
                st.markdown(answer if answer else "_(No answer returned)_")

                sources_text = format_sources(retrieved_docs)
                if sources_text:
                    with st.expander("Sources"):
                        st.text(sources_text)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.append((prompt, answer))

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error: {e}")
