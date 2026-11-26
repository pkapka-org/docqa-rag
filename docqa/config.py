"""All tunable settings live here. Every value can be overridden from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5:1.5b",
    "extractive": "none",
}


@dataclass
class Settings:
    # Chunking. Sizes are in characters; 900 characters is roughly 200 tokens,
    # comfortably inside the embedding model's 512-token window.
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "900")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))

    # Retrieval
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    # Cosine similarity the best chunk must reach before we try to answer at all.
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.60")))

    # Generation
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "extractive").lower())
    model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "4096")))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))

    # Storage (CLI only; the web app keeps each visitor's documents in memory)
    db_path: str = field(default_factory=lambda: _env("DB_PATH", ".chroma"))
    max_file_mb: int = field(default_factory=lambda: int(_env("MAX_FILE_MB", "20")))

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")
