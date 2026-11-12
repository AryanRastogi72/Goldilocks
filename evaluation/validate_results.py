"""
validate_results.py
Recomputes every ranking metric in evaluation/results.json from the .trec run files and the qrels alone,
and checks the run files. Fails (exit code 1) on any difference above 1e-9.

Topics that returned nothing have no lines in a .trec file; results.json records them with retrieved = 0,
and the validator requires that the topics in the file plus those empty topics are exactly the 50 topics.

Usage: python -m evaluation.validate_results
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import compute_set_metrics
from evaluation.run_eval import RESULTS_PATH, score_ranking, trec_path
from evaluation.quick_rank_eval import load_qrels

TOLERANCE = 1e-9
EXPECTED_TOPICS = 50
RANKING_METRICS = ["p_at_10", "ndcg_at_10", "mrr_at_10", "map_at_100", "recall_at_100", "recall_at_1000"]


def read_trec(path):
    """Returns {topic: [(ext_id, rank, score), ...]} in file order. Raises ValueError on a malformed line."""
    runs = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) != 6 or parts[1] != "Q0":
                raise ValueError(f"{path} line {number}: expected 'topic Q0 doc rank score tag'")
            runs[parts[0]].append((parts[2], int(parts[3]), float(parts[4])))
    return runs


def check_run(run_name, field_name, field, qrels, problems):
    path = trec_path(run_name, field_name)
    runs = read_trec(path)
    per_topic = field["per_topic"]
    empty_topics = {topic for topic, record in per_topic.items() if record["retrieved"] == 0}
    if len(set(runs) | empty_topics) != EXPECTED_TOPICS or set(runs) & empty_topics:
        problems.append(f"{path}: {len(runs)} topics in file plus {len(empty_topics)} empty topics, expected {EXPECTED_TOPICS} in total")
    for topic, rows in runs.items():
        ids = [ext_id for ext_id, _, _ in rows]
        if len(ids) != len(set(ids)):
            problems.append(f"{path} topic {topic}: duplicate documents")
        if [rank for _, rank, _ in rows] != list(range(1, len(rows) + 1)):
            problems.append(f"{path} topic {topic}: ranks are not 1..n in file order")
    for topic, record in per_topic.items():
        ids = [ext_id for ext_id, _, _ in runs.get(topic, [])]
        if len(ids) != record["retrieved"]:
            problems.append(f"{path} topic {topic}: {len(ids)} lines but retrieved = {record['retrieved']}")
        recomputed = score_ranking(ids, qrels[topic])
        for name in RANKING_METRICS:
            if abs(recomputed[name] - record[name]) > TOLERANCE:
                problems.append(f"{path} topic {topic} {name}: file gives {recomputed[name]:.12f}, results.json has {record[name]:.12f}")
        if "hits" in record and record["hits"] == len(ids):
            relevant = {d for d, g in qrels[topic].items() if g > 0}
            for name, value in compute_set_metrics(set(ids), relevant).items():
                if abs(value - record[name]) > TOLERANCE:
                    problems.append(f"{path} topic {topic} {name}: file gives {value:.12f}, results.json has {record[name]:.12f}")


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    qrels = load_qrels()
    problems, files = [], 0
    for run_name, run in results["runs"].items():
        for field_name, field in run["fields"].items():
            check_run(run_name, field_name, field, qrels, problems)
            files += 1
    if problems:
        print(f"FAILED: {len(problems)} problems")
        for problem in problems[:30]:
            print("  " + problem)
        sys.exit(1)
    print(f"OK: {files} run files, 50 topics each, all per-topic ranking metrics reproduce within {TOLERANCE}")


if __name__ == "__main__":
    main()
