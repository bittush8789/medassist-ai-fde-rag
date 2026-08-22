import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ingestion.cleaner import clean_medical_text

logger = logging.getLogger(__name__)


class DocumentPage:
    """Represents a single extracted page from a medical document (PDF or TXT)."""
    def __init__(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page_number: int,
        total_pages: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.document_id = document_id
        self.document_name = document_name
        self.page_number = page_number
        self.total_pages = total_pages
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "metadata": self.metadata,
        }


class MedicalDocumentLoader:
    """
    Extracts structured text from medical documents (.pdf and .txt).
    Preserves page boundaries, document metadata, and normalizes text layout.
    """

    def __init__(self, min_page_char_count: int = 15):
        self.min_page_char_count = min_page_char_count

    def load_file(self, file_path: str, custom_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentPage]:
        """Loads a single medical document (.pdf or .txt)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(file_path, custom_metadata=custom_metadata)
        elif suffix in [".txt", ".text", ".md"]:
            return self.load_txt(file_path, custom_metadata=custom_metadata)
        else:
            logger.warning(f"Unsupported file format '{suffix}' for file: {file_path}")
            return []

    def load_pdf(self, file_path: str, custom_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentPage]:
        """Loads a PDF file and returns a list of DocumentPage objects."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for PDF text extraction. Install with `pip install pymupdf`.")

        document_name = path.name
        document_id = path.stem.lower().replace(" ", "_").replace("-", "_")

        pages: List[DocumentPage] = []

        try:
            doc = fitz.open(str(path))
            total_pages = len(doc)
            logger.info(f"Extracting '{document_name}' ({total_pages} pages)...")

            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = doc.load_page(page_idx)
                raw_text = page.get_text("text")

                cleaned_text = clean_medical_text(raw_text)

                if len(cleaned_text) < self.min_page_char_count:
                    logger.warning(
                        f"Page {page_num} in '{document_name}' has fewer than {self.min_page_char_count} chars; "
                        f"it may be empty, an image, or a scanned page."
                    )

                page_meta = {
                    "document_id": document_id,
                    "document_name": document_name,
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "file_path": str(path.resolve()),
                    **(custom_metadata or {})
                }

                pages.append(
                    DocumentPage(
                        text=cleaned_text,
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_num,
                        total_pages=total_pages,
                        metadata=page_meta,
                    )
                )

            doc.close()
            logger.info(f"Successfully processed {len(pages)} pages from '{document_name}'.")
            return pages

        except Exception as e:
            logger.error(f"Failed to process PDF '{file_path}': {str(e)}")
            raise

    def load_txt(self, file_path: str, custom_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentPage]:
        """
        Loads a .txt medical document file.
        Supports explicit page divider tags (e.g. '--- PAGE 2 ---', '[PAGE 2]', '=== PAGE 2 ===')
        or divides long text into logical sections.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"TXT file not found: {file_path}")

        document_name = path.name
        document_id = path.stem.lower().replace(" ", "_").replace("-", "_")

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw_content = f.read()

            cleaned_text = clean_medical_text(raw_content)
            if not cleaned_text:
                logger.warning(f"TXT file '{document_name}' is empty.")
                return []

            # Check for explicit page separators (e.g., "--- PAGE 2 ---", "[PAGE 2]", "=== Page 2 ===")
            page_pattern = r'(?:^|\n)(?:[-=]{3,}\s*(?:PAGE|Page)\s*(\d+)\s*[-=]{3,}|\[(?:PAGE|Page)\s*(\d+)\])(?:\n|$)'
            splits = re.split(page_pattern, cleaned_text)

            pages: List[DocumentPage] = []

            if len(splits) > 1:
                # Document has explicit page demarcations
                page_dict = {}
                first_part = splits[0].strip() if splits[0] else ""
                current_page_num = 1
                
                i = 1
                while i < len(splits):
                    p1 = splits[i]
                    p2 = splits[i+1] if i + 1 < len(splits) else None
                    text_chunk = splits[i+2] if i + 2 < len(splits) else ""
                    
                    matched_num = p1 or p2
                    if matched_num is not None and str(matched_num).strip().isdigit():
                        current_page_num = int(matched_num)
                    
                    if text_chunk and text_chunk.strip():
                        if current_page_num in page_dict:
                            page_dict[current_page_num] += "\n\n" + text_chunk.strip()
                        else:
                            page_dict[current_page_num] = text_chunk.strip()
                    i += 3

                # Prepend top header banner to page 1 if present
                if first_part and 1 in page_dict:
                    page_dict[1] = first_part + "\n\n" + page_dict[1]
                elif first_part and not page_dict:
                    page_dict[1] = first_part

                total_pages = max(page_dict.keys()) if page_dict else 1
                for page_num in sorted(page_dict.keys()):
                    text_chunk = page_dict[page_num]
                    page_meta = {
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "file_path": str(path.resolve()),
                        **(custom_metadata or {})
                    }
                    pages.append(
                        DocumentPage(
                            text=text_chunk,
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_num,
                            total_pages=total_pages,
                            metadata=page_meta,
                        )
                    )
            else:
                # Single page / natural text document or divided every 2500 chars if very long
                max_chars_per_page = 2500
                if len(cleaned_text) <= max_chars_per_page:
                    page_meta = {
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": 1,
                        "total_pages": 1,
                        "file_path": str(path.resolve()),
                        **(custom_metadata or {})
                    }
                    pages.append(
                        DocumentPage(
                            text=cleaned_text,
                            document_id=document_id,
                            document_name=document_name,
                            page_number=1,
                            total_pages=1,
                            metadata=page_meta,
                        )
                    )
                else:
                    # Partition by paragraphs up to max_chars_per_page
                    paragraphs = cleaned_text.split("\n\n")
                    cur_page_text = []
                    cur_len = 0
                    temp_pages = []

                    for p in paragraphs:
                        if cur_len + len(p) > max_chars_per_page and cur_page_text:
                            temp_pages.append("\n\n".join(cur_page_text))
                            cur_page_text = [p]
                            cur_len = len(p)
                        else:
                            cur_page_text.append(p)
                            cur_len += len(p)

                    if cur_page_text:
                        temp_pages.append("\n\n".join(cur_page_text))

                    total_pages = len(temp_pages)
                    for p_idx, p_text in enumerate(temp_pages):
                        page_meta = {
                            "document_id": document_id,
                            "document_name": document_name,
                            "page_number": p_idx + 1,
                            "total_pages": total_pages,
                            "file_path": str(path.resolve()),
                            **(custom_metadata or {})
                        }
                        pages.append(
                            DocumentPage(
                                text=p_text.strip(),
                                document_id=document_id,
                                document_name=document_name,
                                page_number=p_idx + 1,
                                total_pages=total_pages,
                                metadata=page_meta,
                            )
                        )

            logger.info(f"Successfully processed {len(pages)} pages from TXT '{document_name}'.")
            return pages

        except Exception as e:
            logger.error(f"Failed to process TXT '{file_path}': {str(e)}")
            raise

    def load_directory(
        self,
        directory_path: str,
        custom_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[DocumentPage]:
        """
        Loads all supported documents (.pdf and .txt) from a directory.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        files = sorted(list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.txt")))
        if not files:
            logger.warning(f"No PDF or TXT files found in directory: {directory_path}")
            return []

        all_pages: List[DocumentPage] = []
        for file in files:
            custom_meta = (custom_metadata_map or {}).get(file.name, {})
            pages = self.load_file(str(file), custom_metadata=custom_meta)
            all_pages.extend(pages)

        return all_pages


# Backwards compatibility alias
MedicalPDFLoader = MedicalDocumentLoader
