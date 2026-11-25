import hashlib
import re

import pytest
from langchain_core.embeddings import Embeddings

from docqa.config import Settings
from docqa.store import DocumentStore


class HashEmbeddings(Embeddings):
    """Bag-of-words vectors: deterministic, offline, and good enough to rank by shared words."""

    def _vector(self, text):
        v = [1e-6] * 512
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 512] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def embeddings():
    return HashEmbeddings()


@pytest.fixture
def store(embeddings):
    return DocumentStore(embeddings)


@pytest.fixture
def settings():
    return Settings(min_relevance=0.1, top_k=3, provider="extractive", model="")
