"""Streamlit web interface. Run with: streamlit run app.py"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from docqa.config import DEFAULT_MODELS, Settings
from docqa.llm import LLMError, get_llm
from docqa.loaders import LoadError
from docqa.pipeline import Answer, DocumentQA
from docqa.store import DocumentStore, get_embeddings

load_dotenv()
SAMPLES = Path(__file__).parent / "sample_docs"
EXAMPLES = [
    "How many days of annual leave do I get after six years?",
    "Does the warranty cover a cartridge that froze?",
    "What do standard and express delivery cost?",
    "What is the dress code at the office?",
]
LABELS = {
    "extractive": "No LLM (quotes only, free)",
    "anthropic": "Claude (Anthropic)",
    "openai": "GPT (OpenAI)",
    "ollama": "Ollama (local)",
}
KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model…")
def embeddings(name: str):
    return get_embeddings(name)


def safe(text: str) -> str:
    """Stop document text from being read as LaTeX, HTML or an image link."""
    return re.sub(r"([$<>!])", r"\\\1", text)


def show(answer: Answer) -> None:
    if not answer.found:
        st.info(answer.text)
        if answer.reason:
            st.caption(answer.reason)
        return
    st.markdown(safe(answer.text))
    if answer.corrected:
        st.caption("Some citations were added or moved by the automatic check.")
    for c in answer.citations:
        with st.expander(f"[{c.number}] {c.source}, {c.location} · similarity {c.score:.2f}"):
            st.text(c.text)


settings = Settings()
state = st.session_state
if "store" not in state:
    state.store = DocumentStore(embeddings(settings.embedding_model))
    state.history = []
    state.uploader = 0

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT or Markdown", type=["pdf", "txt", "md"],
        accept_multiple_files=True, key=f"upload-{state.uploader}",
    )
    load_samples = st.button("Load the sample documents", width="stretch")

    files = [(u.name, u.getvalue()) for u in uploads or []]
    if load_samples:
        files += [(p.name, p.read_bytes()) for p in sorted(SAMPLES.iterdir())]
    indexer = DocumentQA(state.store, None, settings)
    for name, data in files:
        try:
            with st.spinner(f"Reading {name}…"):
                indexer.add_bytes(name, data)
        except LoadError as exc:
            st.error(str(exc))

    for name, count in state.store.sources().items():
        left, right = st.columns([5, 1])
        left.write(f"{name}  \n:gray[{count} chunks]")
        if right.button("", icon=":material/close:", key=f"remove-{name}", help=f"Remove {name}"):
            state.store.remove(name)
            state.uploader += 1  # reset the uploader so the file is not added straight back
            st.rerun()

    st.header("Answering")
    providers = list(LABELS)
    default = settings.provider if settings.provider in providers else "extractive"
    settings.provider = st.selectbox(
        "Model", providers, index=providers.index(default), format_func=LABELS.get
    )
    api_key = None
    if settings.provider != "extractive":
        settings.model = st.text_input(
            "Model name", value=settings.model or DEFAULT_MODELS[settings.provider],
            key=f"model-{settings.provider}",
        )
    if key_name := KEY_NAMES.get(settings.provider):
        has_key = bool(os.getenv(key_name))
        api_key = st.text_input(
            "API key", type="password", key=f"key-{settings.provider}",
            placeholder="using the server's key" if has_key else "paste your key",
            help="Kept only for this session and never written to disk.",
        ) or None
    if state.history and st.button("Clear the chat", width="stretch"):
        state.history = []
        st.rerun()

st.title("Document Q&A")
st.caption("Answers come only from your documents, and every answer shows its source.")

has_docs = bool(state.store.sources())
if not has_docs:
    st.info("Upload a document in the sidebar, or load the sample documents to try it.")

picked = None
if has_docs and not state.history:
    st.write("Try one of these with the sample documents:")
    for column, example in zip(st.columns(2) * 2, EXAMPLES, strict=True):
        if column.button(example, width="stretch"):
            picked = example

for past in state.history:
    with st.chat_message("user"):
        st.markdown(safe(past.question))
    with st.chat_message("assistant"):
        show(past)

typed = st.chat_input("Ask about your documents", disabled=not has_docs, max_chars=1000)
if question := (typed or picked):
    with st.chat_message("user"):
        st.markdown(safe(question))
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching…"):
                qa = DocumentQA(state.store, get_llm(settings, api_key), settings)
                result = qa.ask(question)
        except LLMError as exc:
            st.error(f"{exc} Check the model name and the API key.")
        else:
            show(result)
            state.history.append(result)
