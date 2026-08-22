import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from backend.config import settings

logger = logging.getLogger(__name__)


class BGEReranker:
    """
    Cross-Encoder Reranker using BGE Reranker models (e.g. BAAI/bge-reranker-base).
    Computes cross-attention relevance scores between query and candidate text chunks.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.embedding_device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder reranker model '{self.model_name}' on device '{self.device}'...")
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Successfully loaded reranker '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder '{self.model_name}': {str(e)}")
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved chunks against the query.
        
        Args:
            query: User's medical query.
            chunks: List of dictionaries with 'content' and 'metadata'.
            top_n: Number of top chunks to return.
            
        Returns:
            List of top_n chunks sorted by reranker score, with 'rerank_score' added.
        """
        if not chunks:
            return []

        if len(chunks) <= top_n and self.model is None:
            return chunks

        if self.model is None:
            # Fallback to original vector similarity ordering
            logger.warning("Reranker model unavailable. Falling back to vector similarity ranking.")
            return chunks[:top_n]

        try:
            pairs = [[query, chunk["content"]] for chunk in chunks]
            raw_scores = self.model.predict(pairs)

            # Apply sigmoid to convert logits to [0, 1] probability range
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))

            scores = [float(sigmoid(s)) if isinstance(s, (float, int, np.floating)) else float(sigmoid(s[0])) for s in raw_scores]

            # Attach scores to chunks
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_copy = dict(chunk)
                chunk_copy["rerank_score"] = round(scores[i], 4)
                scored_chunks.append(chunk_copy)

            # Sort descending by reranker score
            scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            top_results = scored_chunks[:top_n]

            logger.info(
                f"Reranked {len(chunks)} candidate chunks to top {len(top_results)}. "
                f"Top score: {top_results[0]['rerank_score'] if top_results else 'N/A'}"
            )
            return top_results

        except Exception as e:
            logger.error(f"Reranking encountered an error: {str(e)}. Falling back to initial retrieval.")
            return chunks[:top_n]
