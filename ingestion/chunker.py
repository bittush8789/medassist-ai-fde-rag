import re
import uuid
import logging
from typing import List, Dict, Any, Optional
from ingestion.loader import DocumentPage

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunked segment of text with complete metadata inheritance."""
    def __init__(
        self,
        chunk_id: str,
        content: str,
        document_id: str,
        document_name: str,
        page_number: int,
        total_pages: int,
        section: str = "General",
        chunk_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.document_id = document_id
        self.document_name = document_name
        self.page_number = page_number
        self.total_pages = total_pages
        self.section = section
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Returns flattened metadata dictionary compatible with ChromaDB."""
        base_meta = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": int(self.page_number),
            "total_pages": int(self.total_pages),
            "section": str(self.section),
            "chunk_index": int(self.chunk_index),
        }
        # Flatten any additional metadata
        for k, v in self.metadata.items():
            if k not in base_meta:
                if isinstance(v, (str, int, float, bool)):
                    base_meta[k] = v
                else:
                    base_meta[k] = str(v)
        return base_meta


class MedicalChunker:
    """
    Semantic-aware text chunker designed for medical documents.
    Splits text along logical paragraph and sentence boundaries while preserving full citation metadata.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " "]

    def _detect_section(self, text: str, default_section: str = "General") -> str:
        """Heuristically detects section headers within medical document text."""
        header_patterns = [
            r'^(?:SECTION|Section)\s*\d*[:\.\-]?\s*([A-Za-z0-9\s\-]+)',
            r'^#{1,3}\s+([A-Za-z0-9\s\-]+)',
            r'^([A-Z][A-Za-z\s]{3,30}):',
            r'^([0-9]+\.[0-9]*\s+[A-Za-z\s]+)',
        ]
        for line in text.splitlines()[:5]:
            line = line.strip()
            for pattern in header_patterns:
                match = re.match(pattern, line)
                if match:
                    section_name = match.group(1).strip()
                    if 3 <= len(section_name) <= 50:
                        return section_name
        return default_section

    def _split_text(self, text: str) -> List[str]:
        """Recursively splits text into chunks of target size with overlap."""
        if not text or len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Find split point using separators in descending preference
            split_pos = -1
            chunk_slice = text[start:end]

            for sep in self.separators:
                pos = chunk_slice.rfind(sep)
                if pos != -1 and pos > (self.chunk_size // 4):
                    split_pos = start + pos + len(sep)
                    break

            if split_pos == -1 or split_pos <= start:
                split_pos = end

            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Advance start with overlap
            start = max(start + 1, split_pos - self.chunk_overlap)

        return chunks

    def chunk_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """
        Chunks a list of DocumentPage objects into DocumentChunk objects with inherited metadata.
        """
        all_chunks: List[DocumentChunk] = []
        seen_chunk_ids = set()

        for page in pages:
            if not page.text.strip():
                continue

            section = self._detect_section(page.text, default_section="General Overview")
            text_splits = self._split_text(page.text)

            # Determine RBAC and Tenant classification based on document name
            doc_name_lower = page.document_name.lower()
            if "customer_001" in doc_name_lower:
                tenant_id = "customer_001"
                classification = "customer"
                access_roles = "ADMIN,FDE_ENGINEER,CUSTOMER"
                document_type = "customer_guideline"
            elif "customer_002" in doc_name_lower:
                tenant_id = "customer_002"
                classification = "customer"
                access_roles = "ADMIN,FDE_ENGINEER,CUSTOMER"
                document_type = "customer_guideline"
            elif "internal" in doc_name_lower or "runbook" in doc_name_lower:
                tenant_id = "internal"
                classification = "internal"
                access_roles = "ADMIN,FDE_ENGINEER"
                document_type = "technical_runbook"
            else:
                tenant_id = "all"
                classification = "public"
                access_roles = "ADMIN,FDE_ENGINEER,CUSTOMER"
                document_type = "clinical_guideline"

            rbac_meta = {
                **(page.metadata or {}),
                "tenant_id": tenant_id,
                "classification": classification,
                "access_roles": access_roles,
                "document_type": document_type,
            }

            for idx, split_text in enumerate(text_splits):
                base_id = f"{page.document_id}_p{page.page_number}_c{idx + 1}"
                chunk_id = base_id
                suffix = 1
                while chunk_id in seen_chunk_ids:
                    suffix += 1
                    chunk_id = f"{base_id}_{suffix}"
                seen_chunk_ids.add(chunk_id)

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=split_text,
                    document_id=page.document_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    total_pages=page.total_pages,
                    section=section,
                    chunk_index=idx + 1,
                    metadata=rbac_meta,
                )
                all_chunks.append(chunk)

        logger.info(f"Chunked {len(pages)} pages into {len(all_chunks)} semantic chunks with RBAC metadata.")
        return all_chunks
