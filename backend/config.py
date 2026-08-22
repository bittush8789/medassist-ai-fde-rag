from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""
    
    # LLM Settings (Groq)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.1, alias="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=1024, alias="GROQ_MAX_TOKENS")

    # Embedding Settings
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    # Reranker Settings
    reranker_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANKER_MODEL")
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")
    top_k_retrieval: int = Field(default=10, alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=4, alias="TOP_K_RERANK")
    similarity_threshold: float = Field(default=0.35, alias="SIMILARITY_THRESHOLD")
    rerank_threshold: float = Field(default=0.60, alias="RERANK_THRESHOLD")

    # Vector Store Settings
    chroma_persist_directory: str = Field(default="chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="medical_knowledge_base", alias="CHROMA_COLLECTION_NAME")

    # SQLite Database
    database_url: str = Field(default="sqlite:///./medical_chat.db", alias="DATABASE_URL")

    # Observability (LangSmith - Optional)
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="medical-rag-chatbot", alias="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT")

    # Server Configuration
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Security & JWT Authentication
    jwt_secret: str = Field(default="medrag-super-secret-enterprise-jwt-key-2026", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=1440, alias="JWT_EXPIRATION_MINUTES")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
