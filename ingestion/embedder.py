import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """
    HuggingFace BGE Embedding model wrapper.
    Optimized for high retrieval performance and cosine similarity ranking.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Successfully loaded '{self.model_name}'.")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates dense embeddings for a list of document chunks."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generates dense embedding for a user query.
        BGE models can optionally use query instructions for asymmetric retrieval.
        """
        # BGE recommended query instruction for asymmetric retrieval
        instruction = "Represent this sentence for searching relevant passages: "
        formatted_query = f"{instruction}{query}" if "bge" in self.model_name.lower() else query
        embedding = self.model.encode(
            formatted_query,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.tolist()
