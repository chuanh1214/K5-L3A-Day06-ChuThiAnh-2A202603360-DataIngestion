"""Deterministic page-local chunks with source offsets and token safeguards."""
from collections.abc import Callable
import hashlib
import json
import re

from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, start: int, end: int, method: str,
           offset: int = 0) -> Chunk:
    while start < end and page.text[start].isspace():
        start += 1
    while end > start and page.text[end - 1].isspace():
        end -= 1
    text = page.text[start:end]
    metadata = {**page.metadata, "start_char": offset + start,
                "end_char": offset + end, "chunking": method}
    identity = json.dumps([page.document_id, page.source, page.page, text, metadata],
                          ensure_ascii=False, sort_keys=True)
    return Chunk(hashlib.sha256(identity.encode()).hexdigest()[:32],
                 page.document_id, page.source, page.page, text, metadata)


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size.")
    chunks = []
    for start in range(0, len(page.text), size - overlap):
        end = min(start + size, len(page.text))
        chunk = _chunk(page, start, end, "fixed")
        if chunk.text:
            chunks.append(chunk)
        if end == len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack lines/paragraphs; prefer heading and line boundaries over hard cuts.

    The extracted PDF has few blank paragraphs, so headings and line endings are
    the available structure. This does not reconstruct missing code indentation.
    """
    if max_size <= 0:
        raise ValueError("max_size must be positive.")
    chunks = []
    start = end = 0
    heading = re.compile(r"^(?:Chapter\s+\d+|\d+(?:\.\d+)+(?:\s|[A-Za-z]))")
    for line in page.text.splitlines(keepends=True):
        next_end = end + len(line)
        if end > start and (not line.strip() or heading.match(line.strip())
                            or next_end - start > max_size):
            chunk = _chunk(page, start, end, "paragraph_line")
            if chunk.text:
                chunks.append(chunk)
            start = end
        while next_end - start > max_size:
            boundary = page.text.rfind(" ", start + 1, start + max_size + 1)
            cut = boundary if boundary > start else start + max_size
            chunk = _chunk(page, start, cut, "paragraph_line")
            if chunk.text:
                chunks.append(chunk)
            start = cut
        end = next_end
    if end > start:
        chunk = _chunk(page, start, end, "paragraph_line")
        if chunk.text:
            chunks.append(chunk)
    return chunks


def fit_token_limit(chunks: list[Chunk], token_count: Callable[[str], int],
                    max_tokens: int) -> list[Chunk]:
    """Apply the same token guard to both strategies without dropping text."""
    if max_tokens < 3:
        raise ValueError("Token limit must leave room for special tokens and text.")
    result = []
    for chunk in chunks:
        if token_count(chunk.text) <= max_tokens:
            result.append(chunk)
            continue
        page = DocumentPage(chunk.document_id, chunk.source, chunk.page,
                            chunk.text, chunk.metadata)
        start = 0
        while start < len(chunk.text):
            low, high = start + 1, len(chunk.text)
            if token_count(chunk.text[start:low]) > max_tokens:
                raise ValueError("A single character exceeds the token budget.")
            while low < high:
                mid = (low + high + 1) // 2
                if token_count(chunk.text[start:mid]) <= max_tokens:
                    low = mid
                else:
                    high = mid - 1
            cut = low
            if cut < len(chunk.text):
                boundary = chunk.text.rfind(" ", start + 1, cut + 1)
                if boundary > start:
                    cut = boundary
            piece = _chunk(page, start, cut, chunk.metadata["chunking"] + "+token_guard",
                           offset=chunk.metadata["start_char"])
            if piece.text:
                result.append(piece)
            start = cut
            while start < len(chunk.text) and chunk.text[start].isspace():
                start += 1
    return result
