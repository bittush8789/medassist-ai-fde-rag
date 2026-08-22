import pytest
from backend.rag.context import ContextBuilder
from backend.rag.prompts import MEDICAL_RAG_SYSTEM_PROMPT, CONTEXT_PROMPT_TEMPLATE


def test_context_builder_formatting():
    chunks = [
        {
            "chunk_id": "diabetes_p2_c1",
            "content": "Metformin is first-line pharmacotherapy for Type 2 Diabetes.",
            "similarity": 0.88,
            "rerank_score": 0.94,
            "metadata": {
                "document_name": "diabetes_guidelines.pdf",
                "page_number": 2,
                "section": "Pharmacology",
                "chunk_id": "diabetes_p2_c1",
            }
        }
    ]

    formatted_context = ContextBuilder.format_context_for_prompt(chunks)
    assert "DOCUMENT: diabetes_guidelines.pdf" in formatted_context
    assert "PAGE: 2" in formatted_context
    assert "SECTION: Pharmacology" in formatted_context
    assert "Metformin is first-line" in formatted_context

    sources = ContextBuilder.extract_structured_sources(chunks)
    assert len(sources) == 1
    assert sources[0]["document"] == "diabetes_guidelines.pdf"
    assert sources[0]["page"] == 2
    assert sources[0]["relevance_score"] == 0.94


def test_prompt_template_assembly():
    rendered = CONTEXT_PROMPT_TEMPLATE.format(
        context="Sample Context Text",
        history="User: Hello\nAssistant: Hi",
        query="What is the target HbA1c?"
    )
    assert "### RETRIEVED MEDICAL CONTEXT:" in rendered
    assert "Sample Context Text" in rendered
    assert "What is the target HbA1c?" in rendered
