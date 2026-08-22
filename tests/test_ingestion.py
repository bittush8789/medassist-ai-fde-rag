import pytest
import tempfile
from pathlib import Path
from ingestion.cleaner import clean_medical_text
from ingestion.loader import DocumentPage, MedicalDocumentLoader
from ingestion.chunker import MedicalChunker, DocumentChunk


def test_clean_medical_text():
    dirty_text = "  Hyper-\ntension is a major    risk factor.\n\n\n\nPreserve paragraph.\u200b  "
    cleaned = clean_medical_text(dirty_text)
    assert "Hypertension is a major risk factor." in cleaned
    assert "\n\nPreserve paragraph." in cleaned
    assert "\u200b" not in cleaned


def test_chunker_metadata_preservation():
    page = DocumentPage(
        text="Section 1: Pharmacological Treatment of Diabetes\n\nMetformin is the initial drug of choice. Starting dose 500 mg daily.",
        document_id="diabetes_guidelines",
        document_name="diabetes_guidelines.pdf",
        page_number=2,
        total_pages=4,
        metadata={"version": "2024", "author": "Clinical Board"}
    )

    chunker = MedicalChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_name == "diabetes_guidelines.pdf"
        assert chunk.page_number == 2
        assert chunk.total_pages == 4
        assert chunk.chunk_id.startswith("diabetes_guidelines_p2_c")
        meta_dict = chunk.to_metadata_dict()
        assert meta_dict["version"] == "2024"
        assert meta_dict["author"] == "Clinical Board"


def test_txt_document_loader():
    sample_txt_content = """--- PAGE 1 ---
Section 1: Pediatric Otitis Media
High-dose Amoxicillin 80-90 mg/kg/day is the first line treatment.

--- PAGE 2 ---
Section 2: Dehydration Assessment
Mild dehydration corresponds to 3-5% fluid deficit."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(sample_txt_content)
        temp_path = f.name

    try:
        loader = MedicalDocumentLoader()
        pages = loader.load_file(temp_path)
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert "Pediatric Otitis Media" in pages[0].text
        assert pages[1].page_number == 2
        assert "Dehydration Assessment" in pages[1].text
    finally:
        Path(temp_path).unlink(missing_ok=True)
