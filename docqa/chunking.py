"""Split loaded documents into overlapping chunks with stable ids."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
        meta = chunk.metadata
        key = f"{meta.get('source')}:{meta.get('file_hash')}:{index}:{chunk.page_content}"
        chunk.id = hashlib.sha256(key.encode()).hexdigest()[:32]
    return chunks
