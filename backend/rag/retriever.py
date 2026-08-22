import logging
from typing import List, Dict, Any, Optional
from ingestion.chroma_store import ChromaStoreManager
from backend.config import settings

logger = logging.getLogger(__name__)


class MedicalRetriever:
    """
    Retrieval component that fetches semantically relevant medical chunks from ChromaDB.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.store = ChromaStoreManager(
            persist_directory=persist_directory or settings.chroma_persist_directory,
            collection_name=collection_name or settings.chroma_collection_name,
            embedding_model_name=embedding_model_name or settings.embedding_model,
            device=device or settings.embedding_device,
        )

    @staticmethod
    def build_rbac_where_clause(role: Optional[str] = None, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Builds ChromaDB metadata filter based on user role and tenant.
        - ADMIN: Full access (None)
        - FDE_ENGINEER: Internal SOPs + Assigned Tenant + Public docs
        - CUSTOMER: Assigned Tenant + Public docs (Strictly excludes internal docs)
        """
        if not role or role.upper() == "ADMIN":
            return None

        role = role.upper()
        tenant = (tenant_id or "customer_001").lower()

        if role == "FDE_ENGINEER":
            return {
                "$or": [
                    {"tenant_id": tenant},
                    {"tenant_id": "all"},
                    {"classification": "internal"}
                ]
            }
        elif role == "CUSTOMER":
            return {
                "$and": [
                    {"$or": [{"tenant_id": tenant}, {"tenant_id": "all"}]},
                    {"classification": {"$ne": "internal"}}
                ]
            }

        return {"tenant_id": "all"}

    def retrieve(
        self,
        query: str,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
        user_role: Optional[str] = None,
        user_tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K most similar chunks from ChromaDB with RBAC authorization filtering.
        Filters by similarity threshold if specified.
        """
        threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
        
        # Build RBAC where filter if role is supplied and where is not explicitly provided
        active_where = where
        if active_where is None and user_role is not None:
            active_where = self.build_rbac_where_clause(user_role, user_tenant_id)

        results = self.store.similarity_search(query=query, k=k, where=active_where)

        # Filter out chunks below minimum semantic similarity threshold
        filtered_results = [r for r in results if r.get("similarity", 0) >= threshold]
        
        logger.info(
            f"Query: '{query}' (Role: {user_role}, Tenant: {user_tenant_id}, Filter: {active_where}) "
            f"-> Retrieved {len(results)} chunks, {len(filtered_results)} passed similarity threshold ({threshold})."
        )
        return filtered_results if filtered_results else results[:2]  # Fallback to at least top matches if none passed strict threshold
