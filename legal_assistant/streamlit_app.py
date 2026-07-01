"""
BNS 2023 Legal Assistant — Streamlit UI.

Run from project root:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

from legal_assistant.config import BNS_PDF_PATH, GEMINI_API_KEY, INDEX_DIR
from legal_assistant.local_index import index_size
from legal_assistant.rag_pipeline import chat_reply


def _indexed_chunks() -> int:
    return index_size(INDEX_DIR)


st.set_page_config(page_title="BNS Legal Assistant", page_icon="⚖️", layout="wide")

st.title("BNS 2023 — Legal guidelines assistant")
st.caption(
    "Retrieval-augmented answers from your indexed PDF. Not legal advice. "
    "Uses Gemini API + local vector index (sentence-transformers all-MiniLM-L6-v2)."
)

with st.sidebar:
    st.subheader("Index status")
    n = _indexed_chunks()
    st.write(f"Chunks in index: **{n}**")
    st.write(f"PDF path: `{BNS_PDF_PATH}`")
    st.write(f"Index folder: `{INDEX_DIR}`")

    if st.button("Re-ingest PDF (reset index)"):
        with st.spinner("Indexing… this may take a few minutes the first time (downloads embedding model)."):
            r = subprocess.run(
                [sys.executable, "-m", "legal_assistant.ingest", "--reset"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True,
                text=True,
            )
        if r.returncode != 0:
            st.error(r.stderr or r.stdout or "Ingest failed")
        else:
            st.success(r.stdout or "Done")
            st.rerun()

    if n == 0 and st.button("Ingest PDF (first time)"):
        with st.spinner("Indexing…"):
            r = subprocess.run(
                [sys.executable, "-m", "legal_assistant.ingest"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True,
                text=True,
            )
        if r.returncode != 0:
            st.error(r.stderr or r.stdout or "Ingest failed")
        else:
            st.success(r.stdout or "Done")
            st.rerun()

    if not GEMINI_API_KEY:
        st.warning("Set `GEMINI_API_KEY` in `.env` in the project folder.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    role = m.get("role", "user")
    with st.chat_message(role):
        st.markdown(m.get("content", ""))

prompt = st.chat_input("Ask about BNS 2023 or related procedures…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            prior = st.session_state.messages[:-1]
            answer = chat_reply(prior, prompt)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)
