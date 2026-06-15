"""Local filesystem document loader for development and testing without S3."""

import logging
from pathlib import Path

from .document_loader import Document

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS: tuple[str, ...] = (".md", ".txt")


class LocalDocumentLoader:
    """Load Markdown and text files from the local filesystem.

    Drop-in replacement for S3DocumentLoader when running without AWS.
    Same interface: load_all(prefix) returns a list of Documents.
    """

    def __init__(self, docs_dir: str = "docs") -> None:
        """Initialise the loader.

        Args:
            docs_dir: Path to the local directory containing knowledge documents.
        """
        self.docs_dir = Path(docs_dir)

    def load_all(self, prefix: str = "") -> list[Document]:
        """Read every .md and .txt file under docs_dir / prefix.

        Args:
            prefix: Sub-path within docs_dir to filter by (e.g. ``"intro/"``).

        Returns:
            All successfully loaded Documents; unreadable files are skipped.
        """
        search_root = self.docs_dir / prefix if prefix else self.docs_dir
        if not search_root.exists():
            logger.warning("Local docs directory not found: %s", search_root)
            return []

        documents: list[Document] = []
        for path in sorted(search_root.rglob("*")):
            if path.suffix not in _DEFAULT_EXTENSIONS or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
                doc_id = str(path.relative_to(self.docs_dir)).replace("/", "_").replace(".", "_")
                documents.append(
                    Document(
                        content=content,
                        metadata={
                            "source": str(path),
                            "document_id": doc_id,
                        },
                        document_id=doc_id,
                    )
                )
                logger.debug("Loaded local file: %s", path)
            except OSError as e:
                logger.error("Failed to read %s: %s", path, e)

        logger.info("Loaded %d local documents from %s", len(documents), search_root)
        return documents
