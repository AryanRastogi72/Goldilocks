"""
run_eval.py
Runs R0 to R5 on all TREC-COVID topics, both topic fields, to depth 1000, and scores them.

  R0  midsem baseline: AND of the keyword stems, tf-idf ltc.ltc inside the set (set run)
  R1  BM25, full collection
  R2  tf-idf ltc.ltc, full collection
  R3  BM25 + Rocchio pseudo-relevance feedback reranking
  R4  EvidenceLens agent: controller v2.1 (no clarification), BM25 over the whole accepted set (set run)
  R5  R4 with evidence_weight = 0.2 (value fixed before any R5 result existed; nothing else tuned)

Writes evaluation/runs/<run>_<field>.trec (topic Q0 ext_id rank score tag), evaluation/results.json and
evaluation/summary.tsv. Usage: python -m evaluation.run_eval
"""

import json
import os
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.controller import load_config, run_controller
from engine.expansion import rank_bm25_prf
from engine.ranker import rank_bm25, rank_documents
from engine.search import SearchIndex
from evaluation.metrics import (
    compute_map_at_k, compute_mrr_at_k, compute_ndcg_at_k, compute_p_at_k, compute_recall_at_k, compute_set_metrics,
)
from evaluation.quick_rank_eval import FIELDS, load_qrels, load_topics
from pipeline.tokenizer import query_terms, tokenize

DEPTH = 1000
R5_EVIDENCE_WEIGHT = 0.2
RUNS_DIR = os.path.join("evaluation", "runs")
RESULTS_PATH = os.path.join("evaluation", "results.json")
SUMMARY_PATH = os.path.join("evaluation", "summary.tsv")
SET_RUNS = {"R0", "R4", "R5"}
DESCRIPTIONS = {
    "R0": "midsem baseline: AND of keyword stems, tf-idf ltc.ltc inside the set",
    "R1": "BM25, full collection",
    "R2": "tf-idf ltc.ltc, full collection",
    "R3": "BM25 + Rocchio PRF reranking",
    "R4": "EvidenceLens agent (controller v2.1, no clarification), BM25 over the accepted set",
    "R5": f"R4 with evidence_weight {R5_EVIDENCE_WEIGHT}",
}


def midsem_keyword_terms(text):
    """The midsem keyword stems: tokenize and stem, no stopword removal, boolean words dropped, length above 2."""
    return [t for t in tokenize(text) if t not in {"and", "or", "not"} and len(t) > 2]


def run_r0(text, index, config):
    stems = [s for s in midsem_keyword_terms(text) if index.get_df(s) > 0]
    if not stems:
        return [], set(), {}
    docs = index.execute_boolean(("AND", [("TERM", s) for s in stems]))
    return rank_documents(docs, stems, index, top_k=DEPTH), docs, {}


def run_r1(text, index, config):
    ranked = rank_bm25(query_terms(text), index, top_k=DEPTH, k1=config["bm25_k1"], b=config["bm25_b"], title_boost=0.0)
    return ranked, None, {}


def run_r2(text, index, config):
    terms = query_terms(text)
    candidates = {doc_id for term in set(terms) for doc_id, _ in index.get_postings(term)}
    return rank_documents(candidates, terms, index, top_k=DEPTH), None, {}


def run_r3(text, index, config):
    ranked = rank_bm25_prf(
        query_terms(text), index, top_k=DEPTH, k1=config["bm25_k1"], b=config["bm25_b"],
        title_boost=0.0, max_df_fraction=config["max_df_fraction"],
    )
    return ranked, None, {}


def run_agent(text, index, config):
    ranked, log, final_query = run_controller(text, index, config, allow_clarify=False, top_k=DEPTH)
    last = log.steps[-1]
    return ranked, log.final_docs, {"final_query": final_query, "final_decision": last["decision"], "steps": len(log.steps)}


def run_r4(text, index, config):
    return run_agent(text, index, dict(config, evidence_weight=0.0))


def run_r5(text, index, config):
    return run_agent(text, index, dict(config, evidence_weight=R5_EVIDENCE_WEIGHT))


RUN_FUNCTIONS = {"R0": run_r0, "R1": run_r1, "R2": run_r2, "R3": run_r3, "R4": run_r4, "R5": run_r5}


def score_ranking(ranked_ext_ids, topic_qrels):
    relevant = {doc_id for doc_id, grade in topic_qrels.items() if grade > 0}
    return {
        "p_at_10": compute_p_at_k(ranked_ext_ids, relevant, k=10),
        "ndcg_at_10": compute_ndcg_at_k(ranked_ext_ids, topic_qrels, k=10),
        "mrr_at_10": compute_mrr_at_k(ranked_ext_ids, topic_qrels, k=10),
        "map_at_100": compute_map_at_k(ranked_ext_ids, relevant, k=100),
        "recall_at_100": compute_recall_at_k(ranked_ext_ids, relevant, k=100),
        "recall_at_1000": compute_recall_at_k(ranked_ext_ids, relevant, k=1000),
    }


def score_set(set_ext_ids, topic_qrels, config):
    relevant = {doc_id for doc_id, grade in topic_qrels.items() if grade > 0}
    scores = compute_set_metrics(set_ext_ids, relevant)
    scores["hits"] = len(set_ext_ids)
    scores["in_band"] = config["target_band_low"] <= len(set_ext_ids) <= config["target_band_high"]
    return scores


def trec_path(run_name, field_name):
    return os.path.join(RUNS_DIR, f"{run_name}_{field_name.replace('.', '_')}.trec")


def write_trec(path, rankings, tag):
    with open(path, "w", encoding="utf-8") as f:
        for topic_id, scored in rankings.items():
            for rank, (ext_id, score) in enumerate(scored, start=1):
                f.write(f"{topic_id} Q0 {ext_id} {rank} {score:.8f} {tag}\n")


def evaluate_run(run_name, field_name, field_getter, topics, qrels, index, config):
    function = RUN_FUNCTIONS[run_name]
    per_topic, rankings = {}, {}
    for topic in topics:
        topic_id = topic["_id"]
        start = time.perf_counter()
        ranked, doc_set, extra = function(field_getter(topic), index, config)
        latency = time.perf_counter() - start
        scored = [(index.get_ext_id(doc_id), score) for score, doc_id in ranked]
        rankings[topic_id] = scored
        record = score_ranking([ext_id for ext_id, _ in scored], qrels[topic_id])
        record["retrieved"] = len(scored)
        record["latency_s"] = latency
        if run_name in SET_RUNS:
            record.update(score_set({index.get_ext_id(d) for d in doc_set}, qrels[topic_id], config))
        record.update(extra)
        per_topic[topic_id] = record
    write_trec(trec_path(run_name, field_name), rankings, f"{run_name}_{field_name.replace('.', '_')}")
    numeric = [k for k, v in next(iter(per_topic.values())).items() if isinstance(v, (int, float, bool))]
    means = {k: mean(float(r[k]) for r in per_topic.values()) for k in numeric}
    return {"mean": means, "mean_latency_s": means["latency_s"], "per_topic": per_topic}


def write_summary_tsv(results):
    columns = ["p_at_10", "ndcg_at_10", "mrr_at_10", "map_at_100", "recall_at_100", "recall_at_1000", "mean_latency_s",
               "hits", "set_precision", "set_recall", "set_f1", "in_band"]
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\t".join(["run", "field"] + columns) + "\n")
        for run_name, run in results["runs"].items():
            for field_name, field in run["fields"].items():
                values = dict(field["mean"], mean_latency_s=field["mean_latency_s"])
                cells = [f"{values[c]:.4f}" if c in values else "" for c in columns]
                f.write("\t".join([run_name, field_name] + cells) + "\n")


def main():
    started = time.time()
    os.makedirs(RUNS_DIR, exist_ok=True)
    config = load_config()
    index = SearchIndex()
    topics, qrels = load_topics(), load_qrels()
    results = {
        "note": "R5 evidence_weight 0.2 was chosen before any R5 result existed; nothing else was tuned",
        "depth": DEPTH,
        "band": [config["target_band_low"], config["target_band_high"]],
        "runs": {},
    }
    for run_name in RUN_FUNCTIONS:
        kind = "set" if run_name in SET_RUNS else "ranking"
        results["runs"][run_name] = {"description": DESCRIPTIONS[run_name], "kind": kind, "fields": {}}
        for field_name, field_getter in FIELDS.items():
            run_started = time.time()
            field_result = evaluate_run(run_name, field_name, field_getter, topics, qrels, index, config)
            results["runs"][run_name]["fields"][field_name] = field_result
            means = field_result["mean"]
            print(f"{run_name} {field_name:15} nDCG@10={means['ndcg_at_10']:.4f} P@10={means['p_at_10']:.4f} ({time.time() - run_started:.0f}s)", flush=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    write_summary_tsv(results)
    print(f"Saved {RESULTS_PATH}, {SUMMARY_PATH} and 12 run files in {RUNS_DIR}. Total time {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
