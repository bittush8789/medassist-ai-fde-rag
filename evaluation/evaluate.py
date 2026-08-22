import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.retriever import MedicalRetriever
from backend.rag.reranker import BGEReranker
from backend.rag.chain import MedicalRAGPipeline


def run_rag_evaluation(
    dataset_path: str = "evaluation/dataset.json",
    top_k_retrieve: int = 10,
    top_k_rerank: int = 4,
) -> Dict[str, Any]:
    """
    Runs quantitative benchmark evaluation on Medical RAG retrieval and generation.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    retriever = MedicalRetriever()
    reranker = BGEReranker()
    pipeline = MedicalRAGPipeline(retriever=retriever, reranker=reranker)

    results = []
    total_queries = len(dataset)
    retrieval_hits = 0
    rerank_hits = 0
    mrr_sum = 0.0
    negative_rejection_correct = 0
    negative_queries_count = 0
    latencies = []

    print("\n" + "=" * 70)
    print("           MEDICAL RAG SYSTEM BENCHMARK EVALUATION           ")
    print("=" * 70)

    for item in dataset:
        qid = item["id"]
        query = item["question"]
        category = item.get("category", "general")
        exp_doc = item.get("expected_document")
        exp_page = item.get("expected_page")

        start_t = time.time()

        # Step 1: Vector Retrieval
        retrieved_chunks = retriever.retrieve(query=query, k=top_k_retrieve)

        # Step 2: Cross-Encoder Reranking
        if exp_doc is not None:
            reranked_chunks = reranker.rerank(query=query, chunks=retrieved_chunks, top_n=top_k_rerank)
        else:
            reranked_chunks = []

        elapsed_ms = (time.time() - start_t) * 1000
        latencies.append(elapsed_ms)

        # Evaluate Retrieval & Rerank for positive queries
        if exp_doc is not None:
            # Check retrieval hit
            hit_rank = None
            for rank, chunk in enumerate(retrieved_chunks, 1):
                meta = chunk.get("metadata", {})
                if meta.get("document_name") == exp_doc and (exp_page is None or meta.get("page_number") == exp_page):
                    hit_rank = rank
                    break

            is_retrieval_hit = hit_rank is not None
            if is_retrieval_hit:
                retrieval_hits += 1
                mrr_sum += 1.0 / hit_rank

            # Check rerank hit
            rerank_hit = any(
                c.get("metadata", {}).get("document_name") == exp_doc and
                (exp_page is None or c.get("metadata", {}).get("page_number") == exp_page)
                for c in reranked_chunks
            )
            if rerank_hit:
                rerank_hits += 1

            status_str = f"PASS (Rank {hit_rank})" if is_retrieval_hit else "FAIL"
            print(f"[{qid}] Category: {category} -> Retrieval: {status_str} | Rerank Hit: {rerank_hit} ({elapsed_ms:.1f}ms)")

            results.append({
                "id": qid,
                "category": category,
                "query": query,
                "expected": f"{exp_doc} (Page {exp_page})",
                "retrieval_hit": is_retrieval_hit,
                "hit_rank": hit_rank,
                "rerank_hit": rerank_hit,
                "latency_ms": round(elapsed_ms, 2),
            })
        else:
            # Out of scope negative test
            negative_queries_count += 1
            # Run pipeline
            out = pipeline.answer_query(query=query)
            ans = out.get("answer", "").lower()
            is_rejected = "could not find sufficient information" in ans or "not found" in ans or len(out.get("sources", [])) == 0
            if is_rejected:
                negative_rejection_correct += 1
            print(f"[{qid}] Category: {category} -> Negative Refusal: {'PASS (Correctly Refused)' if is_rejected else 'FAIL (Hallucinated)'}")
            results.append({
                "id": qid,
                "category": category,
                "query": query,
                "expected": "Negative / Rejection",
                "is_correctly_rejected": is_rejected,
                "latency_ms": round(elapsed_ms, 2),
            })

    pos_count = total_queries - negative_queries_count
    recall_at_k = (retrieval_hits / pos_count) if pos_count > 0 else 1.0
    rerank_precision = (rerank_hits / pos_count) if pos_count > 0 else 1.0
    mrr = (mrr_sum / pos_count) if pos_count > 0 else 1.0
    neg_accuracy = (negative_rejection_correct / negative_queries_count) if negative_queries_count > 0 else 1.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    summary = {
        "total_queries": total_queries,
        "positive_queries": pos_count,
        "negative_queries": negative_queries_count,
        "recall_at_10": round(recall_at_k, 4),
        "rerank_hit_at_4": round(rerank_precision, 4),
        "mean_reciprocal_rank_mrr": round(mrr, 4),
        "hallucination_refusal_accuracy": round(neg_accuracy, 4),
        "avg_retrieval_latency_ms": round(avg_latency, 2),
        "detailed_results": results,
    }

    print("\n" + "=" * 70)
    print("                    EVALUATION METRICS SUMMARY                ")
    print("=" * 70)
    print(f" • Recall@10 (Dense Vector Search) : {summary['recall_at_10'] * 100:.1f}%")
    print(f" • Top-4 Hit Rate (Cross-Encoder)  : {summary['rerank_hit_at_4'] * 100:.1f}%")
    print(f" • Mean Reciprocal Rank (MRR)      : {summary['mean_reciprocal_rank_mrr']:.4f}")
    print(f" • Zero-Hallucination Rejection    : {summary['hallucination_refusal_accuracy'] * 100:.1f}%")
    print(f" • Average Retrieval Latency       : {summary['avg_retrieval_latency_ms']} ms")
    print("=" * 70 + "\n")

    # Save to json
    out_file = Path("evaluation/evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed results written to {out_file}\n")

    return summary


if __name__ == "__main__":
    run_rag_evaluation()
