import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database.database import init_db
from backend.api.chat import router as chat_router
from backend.api.conversations import router as conv_router
from backend.api.auth import router as auth_router
from backend.api.users import router as users_router
from ingestion.chroma_store import ChromaStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("medical_rag_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown lifecycle."""
    logger.info("Starting up Medical RAG Assistant API...")
    init_db()
    
    # Check if sample PDFs exist
    docs_dir = Path("medical_documents")
    if not docs_dir.exists() or not list(docs_dir.glob("*.pdf")):
        logger.info("Generating sample medical guidelines in 'medical_documents/'...")
        from ingestion.generate_sample_pdfs import generate_all_sample_pdfs
        generate_all_sample_pdfs(str(docs_dir))

    yield
    logger.info("Shutting down Medical RAG Assistant API...")


app = FastAPI(
    title="Medical RAG Assistant API (AI FDE Spec)",
    description="Evidence-grounded conversational AI knowledge assistant with RBAC and multi-tenant isolation.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(chat_router)
app.include_router(conv_router)


@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint returning system status, indexed document count, and configuration state.
    """
    try:
        store_manager = ChromaStoreManager()
        chunk_count = store_manager.count()
        vector_db_status = "healthy"
    except Exception as e:
        chunk_count = 0
        vector_db_status = f"error: {str(e)}"

    has_groq_key = bool(settings.groq_api_key and len(settings.groq_api_key) > 5)

    return {
        "status": "healthy",
        "service": "Medical RAG Assistant API",
        "version": "1.0.0",
        "environment": settings.environment,
        "llm_model": settings.groq_model,
        "groq_configured": has_groq_key,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "use_reranker": settings.use_reranker,
        "vector_store": {
            "status": vector_db_status,
            "indexed_chunks": chunk_count,
            "persist_directory": settings.chroma_persist_directory,
        },
        "langsmith_enabled": bool(settings.langchain_tracing_v2 and settings.langchain_api_key),
    }


# Mount Frontend Static Files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    logger.warning(f"Frontend directory not found at: {frontend_dir}")
