"""Embeddings and the Chroma vector store."""

from __future__ import annotations

import contextlib
import uuid
import weakref
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Runs a small ONNX model on the CPU: no API key, no PyTorch, no data leaves the machine."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=16)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=2)
def get_embeddings(model_name: str) -> Embeddings:
    return LocalEmbeddings(model_name)


def _forget(client, collection: str) -> None:
    with contextlib.suppress(Exception):
        client.delete_collection(collection)


class DocumentStore:
    """A collection of chunks. Pass `path` to keep it on disk; omit it for a private in-memory one."""

    def __init__(self, embeddings: Embeddings, path: str | None = None, collection: str = "documents"):
        options = ChromaSettings(anonymized_telemetry=False)
        if path:
            client = chromadb.PersistentClient(path=path, settings=options)
        else:
            # In-memory Chroma is shared across the whole process, so every store gets
            # its own collection. Without this, web visitors would see each other's files.
            client = chromadb.EphemeralClient(settings=options)
            collection = f"{collection}-{uuid.uuid4().hex[:16]}"
            # Free the memory when the visitor's session, and with it this store, goes away.
            weakref.finalize(self, _forget, client, collection).atexit = False
        self.embeddings = embeddings
        self._db = Chroma(
            client=client,
            collection_name=collection,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Document]) -> int:
        """Add one file's chunks, replacing any older version. Returns how many were stored."""
        if not chunks:
            return 0
        source, file_hash = chunks[0].metadata["source"], chunks[0].metadata["file_hash"]
        existing = self._db.get(where={"source": source}, include=["metadatas"])
        if existing["ids"]:
            if all(m["file_hash"] == file_hash for m in existing["metadatas"]):
                if len(existing["ids"]) == len(chunks):
                    return 0  # same file already indexed
            self._db.delete(ids=existing["ids"])
        self._db.add_documents(chunks, ids=[c.id for c in chunks])
        return len(chunks)

    def remove(self, source: str) -> None:
        ids = self._db.get(where={"source": source}, include=[])["ids"]
        if ids:
            self._db.delete(ids=ids)

    def clear(self) -> None:
        ids = self._db.get(include=[])["ids"]
        if ids:
            self._db.delete(ids=ids)

    def sources(self) -> dict[str, int]:
        """File name -> number of chunks."""
        counts: dict[str, int] = {}
        for meta in self._db.get(include=["metadatas"])["metadatas"]:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return dict(sorted(counts.items()))

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """The k closest chunks with cosine similarity (1.0 = identical), best first."""
        hits = self._db.similarity_search_with_score(query, k=k)
        return [(doc, 1.0 - distance) for doc, distance in hits]
