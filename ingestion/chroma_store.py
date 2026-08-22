import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from ingestion.loader import MedicalPDFLoader
from ingestion.chunker import MedicalChunker, DocumentChunk
from ingestion.embedder import BGEEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class ChromaStoreManager:
    """
    Manages persistent ChromaDB vector store for medical knowledge documents.
    Handles indexing, chunk upsertion, metadata preservation, and similarity search.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "medical_knowledge_base",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = BGEEmbedder(model_name=embedding_model_name, device=device)
        
        # Initialize persistent Chroma client
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Returns total indexed chunk count in the collection."""
        return self.collection.count()

    def reset(self):
        """Clears all vectors in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{self.collection_name}' reset.")

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        """Embeds and upserts document chunks into ChromaDB."""
        if not chunks:
            logger.warning("No chunks provided to index.")
            return 0

        logger.info(f"Indexing {len(chunks)} chunks into ChromaDB collection '{self.collection_name}'...")
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [c.to_metadata_dict() for c in batch]
            embeddings = self.embedder.embed_documents(documents)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            total_added += len(batch)
            logger.info(f"Upserted batch {i // batch_size + 1}: {total_added}/{len(chunks)} chunks.")

        logger.info(f"Successfully indexed total {total_added} chunks.")
        return total_added

    def ingest_documents_directory(
        self,
        docs_dir: str = "medical_documents",
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        reset_existing: bool = False,
    ) -> int:
        """
        End-to-end ingestion pipeline:
        1. Loads all PDFs from docs_dir
        2. Extracts and cleans text per page
        3. Chunks pages into semantic chunks
        4. Indexes into ChromaDB with embeddings
        """
        if reset_existing:
            self.reset()

        loader = MedicalPDFLoader()
        chunker = MedicalChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        pages = loader.load_directory(docs_dir)
        if not pages:
            logger.warning(f"No pages extracted from '{docs_dir}'.")
            return 0

        chunks = chunker.chunk_pages(pages)
        added_count = self.add_chunks(chunks)
        return added_count

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs dense semantic similarity search in ChromaDB.
        Returns retrieved chunks with cosine similarity score and metadata.
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):
                # Convert cosine distance to similarity score: similarity = 1 - distance
                distance = distances[i]
                similarity = 1.0 - distance
                formatted_results.append({
                    "chunk_id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "distance": distance,
                    "similarity": similarity,
                })

        return formatted_results


def run_ingestion_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Medical RAG Document Ingestion CLI")
    parser.add_argument("--docs-dir", default="medical_documents", help="Path to medical PDF directory")
    parser.add_argument("--persist-dir", default="chroma_db", help="Path to ChromaDB persist directory")
    parser.add_argument("--reset", action="store_true", help="Reset existing collection before indexing")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")
    args = parser.parse_args()

    # Generate sample PDFs if directory is empty or missing
    docs_path = Path(args.docs_dir)
    if not docs_path.exists() or not list(docs_path.glob("*.pdf")):
        logger.info("No PDFs found. Generating high-quality medical sample guidelines...")
        from ingestion.generate_sample_pdfs import generate_all_sample_pdfs
        generate_all_sample_pdfs(args.docs_dir)

    manager = ChromaStoreManager(persist_directory=args.persist_dir)
    count = manager.ingest_documents_directory(
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset_existing=args.reset
    )
    print(f"\n[OK] Ingestion complete. Total indexed chunks: {manager.count()}\n")


if __name__ == "__main__":
    run_ingestion_cli()
