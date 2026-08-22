import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from backend.config import settings
from backend.rag.retriever import MedicalRetriever
from backend.rag.reranker import BGEReranker
from backend.rag.prompts import MEDICAL_RAG_SYSTEM_PROMPT, CONTEXT_PROMPT_TEMPLATE
from backend.rag.context import ContextBuilder
from backend.llm.groq_client import GroqLLMClient

logger = logging.getLogger(__name__)


class MedicalRAGPipeline:
    """
    End-to-end Medical RAG Pipeline combining ChromaDB semantic search,
    BGE Cross-Encoder reranking, and Groq LLM generation with citation grounding.
    """

    def __init__(
        self,
        retriever: Optional[MedicalRetriever] = None,
        reranker: Optional[BGEReranker] = None,
        llm_client: Optional[GroqLLMClient] = None,
    ):
        self.retriever = retriever or MedicalRetriever()
        self.reranker = reranker or (BGEReranker() if settings.use_reranker else None)
        self.llm_client = llm_client or GroqLLMClient()
        self.llm = self.llm_client.get_chat_model()

    def _setup_langsmith(self):
        """Configures LangSmith environment variables if enabled."""
        if settings.langchain_tracing_v2 and settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
            logger.info(f"LangSmith Tracing enabled for project: '{settings.langchain_project}'")

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Formats multi-turn conversation messages."""
        if not history:
            return "No previous conversation."
        formatted = []
        for msg in history[-6:]:  # Last 6 turns for context
            role = "User" if msg.get("role") == "user" else "Assistant"
            formatted.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(formatted)

    def answer_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        user_role: Optional[str] = None,
        user_tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a user query with RBAC authorization.
        
        Returns:
            Dict with 'answer', 'sources', 'latency_ms', and 'retrieval_count'.
        """
        start_time = time.time()
        logger.info(f"Processing query: '{query}' (Role: {user_role}, Tenant: {user_tenant_id}) for conv '{conversation_id}'")

        # 1. Retrieval Phase (ChromaDB Vector Search with RBAC filter)
        retrieval_start = time.time()
        candidate_chunks = self.retriever.retrieve(
            query=query,
            k=settings.top_k_retrieval,
            similarity_threshold=settings.similarity_threshold,
            user_role=user_role,
            user_tenant_id=user_tenant_id,
        )
        retrieval_latency = (time.time() - retrieval_start) * 1000

        # If zero relevant chunks found in authorized scope
        if not candidate_chunks:
            logger.info("No authorized chunks retrieved from knowledge base.")
            return {
                "answer": (
                    "I couldn't find relevant information in your authorized knowledge base. "
                    "Please refer to official clinical guidelines or consult a healthcare professional."
                ),
                "sources": [],
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "retrieval_ms": round(retrieval_latency, 2),
                "rerank_ms": 0.0,
                "llm_ms": 0.0,
                "retrieval_count": 0,
                "candidates_count": 0,
                "model": settings.groq_model,
                "rbac_role": user_role,
                "rbac_tenant": user_tenant_id,
            }

        # 2. Reranking Phase (BGE Cross-Encoder)
        rerank_start = time.time()
        if self.reranker and settings.use_reranker:
            reranked = self.reranker.rerank(
                query=query,
                chunks=candidate_chunks,
                top_n=settings.top_k_rerank,
            )
            # Filter out chunks below rerank confidence threshold
            final_chunks = [c for c in reranked if c.get("rerank_score", 1.0) >= settings.rerank_threshold]
        else:
            final_chunks = candidate_chunks[:settings.top_k_rerank]
        rerank_latency = (time.time() - rerank_start) * 1000

        # If all chunks filtered out by reranker confidence threshold
        if not final_chunks:
            logger.info("All chunks rejected by Cross-Encoder confidence threshold.")
            return {
                "answer": (
                    "I could not find sufficient information in the provided medical documents "
                    "to answer this question. Please refer to official clinical guidelines or consult "
                    "a healthcare professional."
                ),
                "sources": [],
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "retrieval_count": 0,
            }

        # 3. Context & Structured Sources Assembly
        context_str = ContextBuilder.format_context_for_prompt(final_chunks)
        history_str = self._format_history(history or [])
        sources = ContextBuilder.extract_structured_sources(final_chunks)

        # 4. Prompt Construction
        prompt_content = CONTEXT_PROMPT_TEMPLATE.format(
            context=context_str,
            history=history_str,
            query=query,
        )

        messages = [
            SystemMessage(content=MEDICAL_RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]

        # 5. LLM Generation (Groq with Automatic Resilient Fallback)
        self._setup_langsmith()
        llm_start = time.time()
        try:
            config = {
                "metadata": {
                    "conversation_id": conversation_id,
                    "retrieved_chunk_count": len(final_chunks),
                    "query": query,
                },
                "tags": ["medical-rag", "groq-llm", settings.environment]
            }
            answer_text = self.llm_client.invoke_with_fallback(messages, config=config)
        except Exception as e:
            logger.error(f"Error during LLM generation: {str(e)}")
            answer_text = (
                "An error occurred while generating the clinical answer. "
                f"Please ensure your `GROQ_API_KEY` is configured properly. (Details: {str(e)[:120]})"
            )
        llm_latency = (time.time() - llm_start) * 1000

        total_latency = (time.time() - start_time) * 1000

        logger.info(
            f"Query processed in {total_latency:.1f}ms "
            f"(Retrieval: {retrieval_latency:.1f}ms, Rerank: {rerank_latency:.1f}ms, LLM: {llm_latency:.1f}ms)"
        )

        return {
            "answer": answer_text,
            "sources": sources,
            "latency_ms": round(total_latency, 2),
            "retrieval_ms": round(retrieval_latency, 2),
            "rerank_ms": round(rerank_latency, 2),
            "llm_ms": round(llm_latency, 2),
            "retrieval_count": len(final_chunks),
            "candidates_count": len(candidate_chunks),
            "model": settings.groq_model,
            "rbac_role": user_role,
            "rbac_tenant": user_tenant_id,
        }
