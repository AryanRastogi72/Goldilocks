"""
quick_rank_eval.py
Full-collection ranking on all TREC-COVID topics: tf-idf cosine vs BM25 vs BM25 + title boost.
Usage: python -m evaluation.quick_rank_eval
"""

import json
import os
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.search import SearchIndex
from engine.ranker import rank_documents, rank_bm25
from pipeline.tokenizer import query_terms
from evaluation.metrics import (
    compute_p_at_k, compute_ndcg_at_k, compute_mrr_at_k, compute_recall_at_k,
)

QUERIES_PATH = os.path.join("data", "raw", "beir", "trec-covid", "queries.jsonl")
QRELS_PATH = os.path.join("data", "cache", "qrels.json")
RESULTS_PATH = os.path.join("evaluation", "quick_rank_results.json")
TOP_K = 100
UNTUNED_BOOST = 0.5
FIELDS = {"metadata.query": lambda topic: topic["metadata"]["query"], "text": lambda topic: topic["text"]}


def load_topics():
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_qrels():
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def tfidf_run(terms, index):
    candidates = {doc_id for term in set(terms) for doc_id, _ in index.get_postings(term)}
    return rank_documents(candidates, terms, index, top_k=TOP_K)


def make_bm25_run(title_boost):
    def run(terms, index):
        return rank_bm25(terms, index, top_k=TOP_K, title_boost=title_boost)
    return run


def score_topic(ranked_ext_ids, topic_qrels):
    relevant = {doc_id for doc_id, grade in topic_qrels.items() if grade > 0}
    return {
        "p_at_10": compute_p_at_k(ranked_ext_ids, relevant, k=10),
        "ndcg_at_10": compute_ndcg_at_k(ranked_ext_ids, topic_qrels, k=10),
        "mrr_at_10": compute_mrr_at_k(ranked_ext_ids, topic_qrels, k=10),
        "recall_at_100": compute_recall_at_k(ranked_ext_ids, relevant, k=100),
    }


def evaluate_run(run, field_getter, topics, qrels, index):
    per_topic = {}
    for topic in topics:
        terms = query_terms(field_getter(topic))
        start = time.time()
        ranked = run(terms, index)
        elapsed = time.time() - start
        ranked_ext_ids = [index.get_ext_id(doc_id) for _, doc_id in ranked]
        per_topic[topic["_id"]] = score_topic(ranked_ext_ids, qrels[topic["_id"]])
        per_topic[topic["_id"]]["query_seconds"] = elapsed
    metric_names = next(iter(per_topic.values())).keys()
    means = {name: mean(row[name] for row in per_topic.values()) for name in metric_names}
    return {"mean": means, "per_topic": per_topic}


def main():
    index = SearchIndex()
    topics = load_topics()
    qrels = load_qrels()
    runs = {
        "tfidf": tfidf_run,
        "bm25": make_bm25_run(0.0),
        f"bm25_title_boost_{UNTUNED_BOOST} (untuned, chosen before seeing results)": make_bm25_run(UNTUNED_BOOST),
    }
    results = {}
    for field_name, field_getter in FIELDS.items():
        results[field_name] = {}
        for run_name, run in runs.items():
            results[field_name][run_name] = evaluate_run(run, field_getter, topics, qrels, index)
            means = results[field_name][run_name]["mean"]
            print(f"{field_name:15} {run_name:75} " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_PATH} ({len(topics)} topics)")


if __name__ == "__main__":
    main()
