import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import docx


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding window word-based chunker."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _parse_pdf(content: bytes) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    doc = fitz.open(stream=content, filetype="pdf")
    raw = doc.metadata
    meta = {
        "author": raw.get("author") or None,
        "title": raw.get("title") or None,
        "created_date": raw.get("creationDate") or None,
        "total_pages": doc.page_count,
    }
    pages = []
    for i in range(doc.page_count):
        text = doc[i].get_text().strip()
        if text:
            pages.append((text, i + 1))
    doc.close()
    return pages, meta


def _parse_docx(content: bytes) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    document = docx.Document(io.BytesIO(content))
    props = document.core_properties
    meta = {
        "author": props.author or None,
        "title": props.title or None,
        "created_date": str(props.created) if props.created else None,
        "total_pages": 1,
    }
    text = "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
    return [(text, 1)], meta


def _parse_txt(content: bytes) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    text = content.decode("utf-8", errors="replace").strip()
    meta = {"author": None, "title": None, "created_date": None, "total_pages": 1}
    return [(text, 1)], meta


def parse_document(
    content: bytes,
    filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse a document into chunks with per-chunk and document-level metadata.

    Returns:
        chunks: list of dicts with keys: text, page_number, chunk_index,
                word_count, char_count
        doc_meta: file-level metadata dict
    """
    ext = Path(filename).suffix.lower().lstrip(".")

    parsers = {"pdf": _parse_pdf, "docx": _parse_docx, "doc": _parse_docx, "txt": _parse_txt}
    if ext not in parsers:
        raise ValueError(f"Unsupported file type: .{ext}  (supported: pdf, docx, txt)")

    pages, doc_meta = parsers[ext](content)

    doc_meta.update(
        {
            "file_type": ext,
            "file_size_bytes": len(content),
            "uploaded_at": datetime.utcnow().isoformat(),
        }
    )

    chunks, idx = [], 0
    for page_text, page_num in pages:
        for chunk in _chunk_text(page_text, chunk_size, chunk_overlap):
            chunks.append(
                {
                    "text": chunk,
                    "page_number": page_num,
                    "chunk_index": idx,
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk),
                }
            )
            idx += 1

    doc_meta["total_chunks"] = len(chunks)
    return chunks, doc_meta
