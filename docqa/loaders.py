"""Turn an uploaded file into LangChain Documents that remember where the text came from.

PDF      -> one Document per page          (metadata: page)
Markdown -> one Document per heading block (metadata: section)
Text     -> one Document
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}


class LoadError(ValueError):
    """The file cannot be used. The message is safe to show to the user."""


def load_path(path: str | Path, max_mb: int = 20) -> list[Document]:
    path = Path(path)
    if not path.is_file():
        raise LoadError(f"File not found: {path}")
    return load_bytes(path.name, path.read_bytes(), max_mb=max_mb)


def load_bytes(name: str, data: bytes, max_mb: int = 20) -> list[Document]:
    name = Path(name).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise LoadError(f"{name}: unsupported file type. Use PDF, TXT or Markdown.")
    if not data:
        raise LoadError(f"{name}: the file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise LoadError(f"{name}: larger than the {max_mb} MB limit.")

    if suffix == ".pdf":
        docs = _load_pdf(name, data)
    elif suffix == ".txt":
        docs = [Document(page_content=_decode(data))]
    else:
        docs = _split_markdown(_decode(data))

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        hint = " It may be a scan; OCR is not supported." if suffix == ".pdf" else ""
        raise LoadError(f"{name}: no readable text found.{hint}")

    file_hash = hashlib.sha256(data).hexdigest()[:16]
    for d in docs:
        d.metadata.update(source=name, file_hash=file_hash)
    return docs


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _load_pdf(name: str, data: bytes) -> list[Document]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise LoadError(f"{name}: the PDF is password protected.")
        return [
            Document(page_content=_tidy(page.extract_text() or ""), metadata={"page": number})
            for number, page in enumerate(reader.pages, start=1)
        ]
    except LoadError:
        raise
    except (PyPdfError, ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        raise LoadError(f"{name}: not a readable PDF.") from exc


def _tidy(text: str) -> str:
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)  # re-join words hyphenated at a line end
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _split_markdown(text: str) -> list[Document]:
    """Split on headings, ignoring '#' lines inside fenced code blocks."""
    docs: list[Document] = []
    trail: list[tuple[int, str]] = []
    buffer: list[str] = []
    fence = ""

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        section = " > ".join(title for _, title in trail)
        # Keep the heading in the text so it is embedded along with the body.
        content = f"{section}\n\n{body}" if section else body
        docs.append(Document(page_content=content, metadata={"section": section} if section else {}))

    for line in text.splitlines():
        stripped = line.lstrip()
        if fence:
            if stripped.startswith(fence):
                fence = ""
        elif stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
        elif match := _HEADING.match(line):
            flush()
            level = len(match[1])
            while trail and trail[-1][0] >= level:
                trail.pop()
            trail.append((level, match[2]))
            continue
        buffer.append(line)
    flush()
    return docs
