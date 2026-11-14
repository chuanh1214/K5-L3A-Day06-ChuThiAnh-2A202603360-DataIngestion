"""Read and normalize source data. TODO: implement functions."""

from pathlib import Path
from .models import DocumentPage
from pypdf import PdfReader
import logging
import re
import unicodedata


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    pages = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        try:
            text = normalize_text(page.extract_text() or "")
        except Exception:
            logging.exception("Extraction failed: %s page=%s", path, number)
            raise
        if not text:
            logging.warning("Empty page: %s page=%s", path, number)
        pages.append(DocumentPage(document_id, source, number, text))
    return pages


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
