import re
from typing import List, Dict, Any, Tuple


class ContextBuilder:
    """
    Constructs clean, structured context strings for LLM prompts
    and extracts citation references from retrieved chunks.
    """

    @staticmethod
    def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
        """
        Formats a list of retrieved/reranked chunks into a structured prompt context block.
        """
        if not chunks:
            return "No relevant medical documents retrieved."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document_name", "Unknown Document")
            page_num = meta.get("page_number", "N/A")
            section = meta.get("section", "General")
            content = chunk.get("content", "").strip()

            block = (
                f"--- [SOURCE {idx}] ---\n"
                f"DOCUMENT: {doc_name}\n"
                f"PAGE: {page_num}\n"
                f"SECTION: {section}\n\n"
                f"{content}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    @staticmethod
    def extract_structured_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts de-duplicated structured source references for frontend display and API responses.
        """
        seen_keys = set()
        sources = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document_name", "Unknown Document")
            page_num = meta.get("page_number", 1)
            section = meta.get("section", "General")
            chunk_id = chunk.get("chunk_id", meta.get("chunk_id", ""))
            relevance = chunk.get("rerank_score", chunk.get("similarity", 0.0))

            key = (doc_name, page_num, section)
            if key not in seen_keys:
                seen_keys.add(key)
                # Snippet preview (first 180 chars)
                snippet = chunk.get("content", "")[:180].strip() + ("..." if len(chunk.get("content", "")) > 180 else "")
                sources.append({
                    "document": doc_name,
                    "page": int(page_num) if str(page_num).isdigit() else page_num,
                    "section": section,
                    "chunk_id": chunk_id,
                    "relevance_score": round(float(relevance), 3) if relevance else None,
                    "snippet": snippet,
                })

        return sources
