import re
from typing import Any
from .base import BaseChunker, Chunk


class MarkdownChunker(BaseChunker):
    """Split markdown documents on header boundaries, preserving hierarchy."""

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 100,
    ):
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self._header_levels = {h: lvl for h, lvl in self.headers_to_split_on}

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        doc_id = metadata.get("document_id", "doc")
        sections = self._split_on_headers(text)
        chunks = []
        index = 0
        for section in sections:
            content = section["content"].strip()
            if not content:
                continue
            # Sub-split oversized sections
            if len(content) > self.max_chunk_size:
                sub_chunks = self._sub_split(content)
            else:
                sub_chunks = [content]

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
        sections: list[dict[str, Any]] = []
        current_content_lines: list[str] = []
        current_header = ""
        current_level = ""
        breadcrumb_stack: list[str] = []

        for line in text.splitlines(keepends=True):
            match = self.HEADER_PATTERN.match(line.rstrip())
            if match:
                hashes, title = match.group(1), match.group(2)
                level = self._header_levels.get(hashes, "")
                if level:
                    # Save previous section
                    if current_content_lines or current_header:
                        sections.append(
                            {
                                "header": current_header,
                                "level": current_level,
                                "content": "".join(current_content_lines),
                                "breadcrumb": " > ".join(breadcrumb_stack),
                            }
                        )
                    # Update breadcrumb
                    depth = len(hashes)
                    breadcrumb_stack = breadcrumb_stack[: depth - 1]
                    breadcrumb_stack.append(title)
                    current_header = title
                    current_level = level
                    current_content_lines = [line]
                    continue
            current_content_lines.append(line)

        if current_content_lines or current_header:
            sections.append(
                {
                    "header": current_header,
                    "level": current_level,
                    "content": "".join(current_content_lines),
                    "breadcrumb": " > ".join(breadcrumb_stack),
                }
            )
        return sections

    def _sub_split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            # Try to break at paragraph boundary
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start + self.chunk_overlap:
                end = newline_pos
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks
