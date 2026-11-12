"""
controller_eval.py
Runs the offline controller on all TREC-COVID topics and compares its ranking with full-collection BM25.
Informative only; the formal comparison is part of the final evaluation phase.
Usage: python -m evaluation.controller_eval
"""

import json
import os
import sys
from collections import Counter
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.controller import load_config, run_controller
from engine.search import SearchIndex
from evaluation.metrics import compute_ndcg_at_k, compute_p_at_k
from evaluation.quick_rank_eval import FIELDS, load_qrels, load_topics

RUNS_PATH = os.path.join("evaluation", "controller_runs.json")
QUICK_PATH = os.path.join("evaluation", "quick_rank_results.json")


def run_topic(topic, field_getter, config, index, qrels):
    ranked, log, final_query = run_controller(field_getter(topic), index, config)
    ranked_ext_ids = [index.get_ext_id(doc_id) for _, doc_id in ranked]
    topic_qrels = qrels[topic["_id"]]
    relevant = {doc_id for doc_id, grade in topic_qrels.items() if grade > 0}
    final_step = log.steps[-1]
    return {
        "topic": topic["_id"],
        "input": field_getter(topic),
        "final_query": final_query,
        "final_hits": final_step["actual_hits"],
        "final_decision": final_step["decision"],
        "in_band": final_step["decision"] == "ACCEPT",
        "p_at_10": compute_p_at_k(ranked_ext_ids, relevant, k=10),
        "ndcg_at_10": compute_ndcg_at_k(ranked_ext_ids, topic_qrels, k=10),
        "trace": log.to_dict(),
    }


def summarize(runs):
    decisions = Counter(step["decision"] for run in runs for step in run["trace"]["steps"])
    return {
        "topics": len(runs),
        "in_band_rate": mean(run["in_band"] for run in runs),
        "mean_steps": mean(len(run["trace"]["steps"]) for run in runs),
        "decision_counts": dict(decisions),
        "mean_p_at_10": mean(run["p_at_10"] for run in runs),
        "mean_ndcg_at_10": mean(run["ndcg_at_10"] for run in runs),
    }


def main():
    config = load_config()
    index = SearchIndex()
    topics, qrels = load_topics(), load_qrels()
    with open(QUICK_PATH, "r", encoding="utf-8") as f:
        quick = json.load(f)

    output = {}
    for field_name, field_getter in FIELDS.items():
        runs = [run_topic(topic, field_getter, config, index, qrels) for topic in topics]
        summary = summarize(runs)
        bm25_means = quick[field_name]["bm25"]["mean"]
        summary["full_collection_bm25_p_at_10"] = bm25_means["p_at_10"]
        summary["full_collection_bm25_ndcg_at_10"] = bm25_means["ndcg_at_10"]
        output[field_name] = {"summary": summary, "runs": runs}
        print(f"\nField: {field_name}")
        for name, value in summary.items():
            print(f"  {name}: {value if not isinstance(value, float) else round(value, 4)}")

    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {RUNS_PATH}")


if __name__ == "__main__":
    main()
