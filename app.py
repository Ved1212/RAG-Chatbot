import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Chat with your PDFs", page_icon="💬")
st.title("💬 Chat with your documents")

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
needs_key = PROVIDER != "ollama"

with st.sidebar:
    st.caption(f"LLM provider: **{PROVIDER}**")
    api_key = ""
    if needs_key:
        api_key = st.text_input("Gemini API key", type="password",
                                value=os.getenv("GEMINI_API_KEY", ""))
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process = st.button("Process documents", type="primary")

from src.chunker import chunk_pages  # noqa: E402
from src.loader import load_pdf  # noqa: E402
from src.rag import answer  # noqa: E402
from src.store import VectorStore  # noqa: E402

if process and files:
    with st.spinner("Reading, chunking and embedding..."):
        pages = [p for f in files for p in load_pdf(f.name, f.getvalue())]
        chunks = chunk_pages(pages)
        store = VectorStore()
        store.add(chunks)
        st.session_state.store = store
        st.session_state.messages = []
    st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(files)} file(s)")

for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask something about your documents"):
    if "store" not in st.session_state:
        st.warning("Upload PDFs and click 'Process documents' first.")
    elif needs_key and not api_key:
        st.warning("Add your Gemini API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    history = st.session_state.messages[:-1]
                    reply, hits = answer(st.session_state.store, q, history)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()
            st.markdown(reply)
            with st.expander("Sources used"):
                for h in hits:
                    st.caption(f"{h['source']} — page {h['page']} (distance {h['distance']:.2f})")
                    st.write(h["text"][:300] + "...")
        st.session_state.messages.append({"role": "assistant", "content": reply})
