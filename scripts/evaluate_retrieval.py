import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config, storage
from backend.index.embeddings import Embedder
from backend.index.reranker import Reranker
from backend.index.sparse_index import SparseIndex
from backend.index.vector_index import VectorIndex
from backend.services.retrieval_service import RetrievalService


def _recall_at_k(results: List[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(results[:k]).intersection(relevant))
    return hits / float(len(relevant))


def _mrr_at_k(results: List[str], relevant: set[str], k: int) -> float:
    for idx, chunk_id in enumerate(results[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / float(idx)
    return 0.0


def _ndcg_at_k(results: List[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, chunk_id in enumerate(results[:k], start=1):
        rel = 1.0 if chunk_id in relevant else 0.0
        if rel > 0:
            dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(2, ideal_hits + 2))
    return dcg / idcg if idcg > 0 else 0.0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / float(len(items)) if items else 0.0


def evaluate(
    eval_set: list[dict],
    retrieval: RetrievalService,
    mode: str,
    k_values: list[int],
    use_rerank: bool,
) -> dict:
    max_k = max(k_values)
    results_by_k = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}
    per_slice = defaultdict(lambda: {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values})
    skipped = 0

    for row in eval_set:
        query = str(row.get("query") or "").strip()
        relevant_ids = {str(x) for x in row.get("relevant_chunk_ids", []) if str(x).strip()}
        if not query or not relevant_ids:
            skipped += 1
            continue
        doc_ids = row.get("doc_ids")
        if doc_ids is not None:
            doc_ids = [str(d) for d in doc_ids]
        hits = retrieval.search(
            query=query,
            top_k=max_k,
            doc_ids=doc_ids,
            mode=mode,
            use_rerank=use_rerank,
        )
        ranked = [str(hit.get("id")) for hit in hits if hit.get("id")]
        slice_name = str(row.get("slice") or "default")
        for k in k_values:
            recall = _recall_at_k(ranked, relevant_ids, k)
            mrr = _mrr_at_k(ranked, relevant_ids, k)
            ndcg = _ndcg_at_k(ranked, relevant_ids, k)
            results_by_k[k]["recall"].append(recall)
            results_by_k[k]["mrr"].append(mrr)
            results_by_k[k]["ndcg"].append(ndcg)
            per_slice[slice_name][k]["recall"].append(recall)
            per_slice[slice_name][k]["mrr"].append(mrr)
            per_slice[slice_name][k]["ndcg"].append(ndcg)

    aggregate = {
        f"@{k}": {
            "recall": round(_mean(stats["recall"]), 4),
            "mrr": round(_mean(stats["mrr"]), 4),
            "ndcg": round(_mean(stats["ndcg"]), 4),
        }
        for k, stats in results_by_k.items()
    }
    slices = {}
    for name, bucket in per_slice.items():
        slices[name] = {
            f"@{k}": {
                "recall": round(_mean(stats["recall"]), 4),
                "mrr": round(_mean(stats["mrr"]), 4),
                "ndcg": round(_mean(stats["ndcg"]), 4),
            }
            for k, stats in bucket.items()
        }
    return {
        "queries_total": len(eval_set),
        "queries_scored": len(eval_set) - skipped,
        "queries_skipped": skipped,
        "mode": mode,
        "use_rerank": use_rerank,
        "metrics": aggregate,
        "slices": slices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on a labeled query set.")
    parser.add_argument("--eval-set", required=True, help="Path to JSON file with retrieval eval rows.")
    parser.add_argument("--mode", default="hybrid", choices=["semantic", "sparse", "hybrid", "image_first"])
    parser.add_argument("--k", default="1,3,5,10", help="Comma-separated k values, e.g. 1,3,5,10")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking during evaluation.")
    args = parser.parse_args()

    load_dotenv()
    config.ensure_dirs()
    storage.init_db()

    k_values = sorted({int(item.strip()) for item in args.k.split(",") if item.strip()})
    eval_path = Path(args.eval_set)
    eval_rows = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(eval_rows, list):
        raise ValueError("Eval set must be a JSON array.")

    embed_model = config.OPENAI_EMBED_MODEL if config.EMBED_PROVIDER == "openai" else config.EMBED_MODEL
    embedder = Embedder(
        embed_model,
        device="cuda",
        provider=config.EMBED_PROVIDER,
        openai_api_key=config.OPENAI_API_KEY,
    )
    vector_index = VectorIndex()
    vector_index.load()
    sparse_index = SparseIndex()
    sparse_index.load()
    reranker = Reranker(
        model_name=config.RERANK_MODEL_ID,
        device=config.RERANK_DEVICE,
        enabled=config.ENABLE_RERANKER and not args.no_rerank,
    )
    retrieval = RetrievalService(
        embedder,
        vector_index,
        sparse_index=sparse_index,
        reranker=reranker,
        default_mode=args.mode,
        rerank_top_n=config.RERANK_TOP_N,
    )

    report = evaluate(
        eval_rows,
        retrieval=retrieval,
        mode=args.mode,
        k_values=k_values,
        use_rerank=not args.no_rerank,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
