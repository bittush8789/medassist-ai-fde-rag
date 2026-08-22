import logging
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from ingestion.embedder import BGEEmbedder
from backend.config import settings

logger = logging.getLogger(__name__)


class LangChainBGEEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings adapter for BGE embedding models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.embedder = BGEEmbedder(
            model_name=self.model_name,
            device=self.device,
            normalize_embeddings=normalize_embeddings,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)
