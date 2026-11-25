import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from docqa.chunking import chunk_documents
from docqa.loaders import LoadError, load_bytes, load_path

HANDBOOK = Path(__file__).parent.parent / "sample_docs" / "employee_handbook.pdf"


def test_pdf_keeps_page_numbers():
    docs = load_path(HANDBOOK)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3, 4, 5, 6]
    assert "26 days" in docs[1].page_content
    assert docs[0].metadata["source"] == "employee_handbook.pdf"


def test_markdown_sections_are_nested_and_code_is_not_a_heading():
    text = b"# A\nintro\n## B\nbody\n```\n# comment\n```\n# C\nend"
    docs = load_bytes("x.md", text)
    assert [d.metadata["section"] for d in docs] == ["A", "A > B", "C"]
    assert "# comment" in docs[1].page_content


def test_text_encodings():
    assert "é" in load_bytes("a.txt", "café".encode("cp1252"))[0].page_content
    assert load_bytes("a.txt", "﻿hi".encode())[0].page_content == "hi"


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("a.exe", b"x", "unsupported"),
        ("a.txt", b"", "empty"),
        ("a.txt", b"  \n ", "no readable text"),
        ("a.pdf", b"not a pdf", "not a readable PDF"),
        ("a.pdf", b"%PDF-1.4\n%%EOF", "not a readable PDF"),
    ],
)
def test_bad_files_give_clear_errors(name, data, message):
    with pytest.raises(LoadError, match=message):
        load_bytes(name, data)


def test_size_limit():
    with pytest.raises(LoadError, match="larger"):
        load_bytes("a.txt", b"x" * (1024 * 1024 + 1), max_mb=1)


def test_pdf_without_text_mentions_scans():
    writer, out = PdfWriter(), io.BytesIO()
    writer.add_blank_page(200, 200)
    writer.write(out)
    with pytest.raises(LoadError, match="scan"):
        load_bytes("blank.pdf", out.getvalue())


def test_password_protected_pdf():
    writer, out = PdfWriter(), io.BytesIO()
    writer.append(PdfReader(HANDBOOK))
    writer.encrypt("secret", algorithm="RC4-128")
    writer.write(out)
    with pytest.raises(LoadError, match="password"):
        load_bytes("locked.pdf", out.getvalue())


def test_path_in_file_name_is_removed():
    assert load_bytes("../../etc/a.txt", b"hello")[0].metadata["source"] == "a.txt"


def test_chunks_are_bounded_and_ids_are_stable_and_unique():
    docs = load_bytes("a.txt", ("word " * 2000).encode())
    first, second = (chunk_documents(docs, 300, 50) for _ in range(2))
    assert all(len(c.page_content) <= 300 for c in first)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
