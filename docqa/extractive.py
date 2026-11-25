"""No-LLM answering: quote the sentences closest to the question, with their citations.

It cannot combine or rephrase, but it costs nothing, needs no key and cannot invent text.
"""

from __future__ import annotations

import re

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n{2,}|\n(?=\s*(?:[-*•]|\d+\.)\s)")


_HEADING_LINE = re.compile(r"^(.{1,60}[^.!?:;,\s])\n(?=[A-Z])", re.MULTILINE)


def split_sentences(text: str) -> list[str]:
    text = _HEADING_LINE.sub(r"\1\n\n", text.replace("**", ""))
    parts = (re.sub(r"\s+", " ", p).strip(" -*•#") for p in _SENTENCE.split(text))
    parts = [p for p in parts if len(p.split()) >= 4]
    # Headings have no closing punctuation. Fall back to everything if nothing has any.
    return [p for p in parts if p[-1] in ".!?"] or parts


def extractive_answer(
    question: str, hits: list[tuple[Document, float]], embeddings: Embeddings, limit: int = 2
) -> str:
    candidates: list[tuple[str, int]] = []
    seen: set[str] = set()
    for number, (doc, _) in enumerate(hits[:3], start=1):
        for sentence in split_sentences(doc.page_content):
            if sentence not in seen:
                seen.add(sentence)
                candidates.append((sentence, number))
    if not candidates:
        return ""

    query = np.array(embeddings.embed_query(question))
    matrix = np.array(embeddings.embed_documents([s for s, _ in candidates]))
    scores = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-9)

    order = np.argsort(-scores)
    # Drop the second sentence when it is clearly weaker than the first.
    best = [i for i in order[:limit] if scores[i] >= scores[order[0]] - 0.06]
    return " ".join(f"{candidates[i][0]} [{candidates[i][1]}]" for i in best)
