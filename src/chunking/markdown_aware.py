"""Markdown-aware chunker that splits on header boundaries and preserves hierarchy."""

import re
from typing import Any

from config.settings import (
    MARKDOWN_MAX_CHUNK_SIZE,
    MARKDOWN_CHUNK_OVERLAP,
    MARKDOWN_HEADERS,
)
from .base import BaseChunker, Chunk


class MarkdownChunker(BaseChunker):
    """Split markdown documents on header boundaries, preserving breadcrumb hierarchy.

    Each ``#``-level header starts a new chunk.  Sections that exceed
    *max_chunk_size* are sub-split at the nearest paragraph boundary.
    Header text and breadcrumb path are stored in chunk metadata so
    retrievers can surface the document hierarchy alongside the content.
    """

    _HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        max_chunk_size: int = MARKDOWN_MAX_CHUNK_SIZE,
        chunk_overlap: int = MARKDOWN_CHUNK_OVERLAP,
    ) -> None:
        """Initialise the chunker.

        Args:
            headers_to_split_on: List of (markdown_prefix, label) pairs, e.g.
                ``[("#", "h1"), ("##", "h2")]``.  Defaults to h1–h4.
            max_chunk_size: Characters above which a section is sub-split.
            chunk_overlap: Characters of overlap when sub-splitting an oversized section.
        """
        self.headers_to_split_on = headers_to_split_on or list(MARKDOWN_HEADERS)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self._header_levels: dict[str, str] = {h: lvl for h, lvl in self.headers_to_split_on}

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* on markdown headers and return one Chunk per section.

        Args:
            text: Markdown document text.
            metadata: Passed through to every produced Chunk; ``section_header``,
                ``header_level``, and ``breadcrumb`` are added per-chunk.

        Returns:
            Ordered list of Chunks, one per markdown section (or sub-section if oversized).
        """
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        sections = self._split_on_headers(text)
        chunks: list[Chunk] = []
        index = 0
        for section in sections:
            content = section["content"].strip()
            if not content:
                continue
            sub_chunks = (
                self._sub_split(content)
                if len(content) > self.max_chunk_size
                else [content]
            )
            for sub in sub_chunks:
                section_meta = {
                    **metadata,
                    "section_header": section.get("header", ""),
                    "header_level": section.get("level", ""),
                    "breadcrumb": section.get("breadcrumb", ""),
                }
                start = text.find(sub)
                chunks.append(
                    self._make_chunk(sub, index, doc_id, start, start + len(sub), section_meta)
                )
                index += 1
        return chunks

    def _split_on_headers(self, text: str) -> list[dict[str, Any]]:
        """Walk the document line-by-line and group lines between headers into sections.

        Args:
            text: Full markdown document.

        Returns:
            List of dicts with keys ``header``, ``level``, ``content``, ``breadcrumb``.
        """
        sections: list[dict[str, Any]] = []
        current_lines: list[str] = []
        current_header = ""
        current_level = ""
        breadcrumb_stack: list[str] = []

        for line in text.splitlines(keepends=True):
            match = self._HEADER_PATTERN.match(line.rstrip())
            if match:
                hashes, title = match.group(1), match.group(2)
                level = self._header_levels.get(hashes, "")
                if level:
                    if current_lines or current_header:
                        sections.append({
                            "header": current_header,
                            "level": current_level,
                            "content": "".join(current_lines),
                            "breadcrumb": " > ".join(breadcrumb_stack),
                        })
                    depth = len(hashes)
                    breadcrumb_stack = breadcrumb_stack[: depth - 1]
                    breadcrumb_stack.append(title)
                    current_header = title
                    current_level = level
                    current_lines = [line]
                    continue
            current_lines.append(line)

        if current_lines or current_header:
            sections.append({
                "header": current_header,
                "level": current_level,
                "content": "".join(current_lines),
                "breadcrumb": " > ".join(breadcrumb_stack),
            })
        return sections

    def _sub_split(self, text: str) -> list[str]:
        """Break an oversized section at paragraph boundaries into smaller pieces.

        Args:
            text: Section content that exceeds *max_chunk_size*.

        Returns:
            List of text fragments each no larger than *max_chunk_size*.
        """
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start + self.chunk_overlap:
                end = newline_pos
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks
