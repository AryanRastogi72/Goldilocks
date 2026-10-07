"""Check saved evaluation metrics against identifiers and qrels."""

import gzip
import csv
import json
import math


def main():
    with open("data/cache/qrels.json", "r", encoding="utf-8") as file:
        qrels = json.load(file)
    with open("data/cache/queries.json", "r", encoding="utf-8") as file:
        queries = json.load(file)
    with open("evaluation/results.json", "r", encoding="utf-8") as file:
        results = json.load(file)
    with gzip.open("evaluation/retrieved_ids.json.gz", "rt", encoding="utf-8") as file:
        retrieved = json.load(file)

    checked = 0
    failures = []
    expected_qids = set(queries) & set(qrels)
    for method, entries in results.items():
        qids = [entry["qid"] for entry in entries]
        if len(qids) != len(set(qids)) or set(qids) != expected_qids:
            failures.append(f"{method}: topic coverage does not match queries and qrels")
        for entry in entries:
            if not entry.get("available", True):
                continue
            qid = entry["qid"]
            relevant = {doc_id for doc_id, score in qrels[qid].items() if score > 0}
            key = entry["retrieved_ids_key"]
            docs = set(retrieved[method][qid])
            ranked = entry["ranked_doc_ids"]
            overlap = len(docs & relevant)
            precision = overlap / len(docs) if docs else 0.0
            recall = overlap / len(relevant) if relevant else 0.0
            p_at_10 = sum(doc_id in relevant for doc_id in ranked[:10]) / 10
            checked += 1
            expected = {
                "precision": precision,
                "recall": recall,
                "p_at_10": p_at_10,
            }
            if key != f"{method}:{qid}" or len(docs) != entry["hits"]:
                failures.append(f"{method} topic {qid}: identifier count or key mismatch")
            for name, value in expected.items():
                if not math.isclose(entry[name], value, rel_tol=0, abs_tol=1e-12):
                    failures.append(f"{method} topic {qid}: {name} does not match saved identifiers")

    summary_path = "evaluation/summary.tsv"
    with open(summary_path, "r", encoding="utf-8", newline="") as file:
        summary_rows = {row["Method"]: row for row in csv.DictReader(file, delimiter="\t")}
    for method, entries in results.items():
        available = [entry for entry in entries if entry.get("available", True)]
        if not available:
            continue
        row = summary_rows.get(method)
        if row is None:
            failures.append(f"{method}: no row in saved summary table")
            continue
        expected_summary = {
            "In Band %": sum(entry["in_band"] for entry in available) * 100 / len(available),
            "Avg Hits": sum(entry["hits"] for entry in available) / len(available),
            "Avg Precision": sum(entry["precision"] for entry in available) / len(available),
            "Avg Recall": sum(entry["recall"] for entry in available) / len(available),
            "Avg P@10": sum(entry["p_at_10"] for entry in available) / len(available),
        }
        for name, value in expected_summary.items():
            tolerance = 0.051 if name in {"In Band %", "Avg Hits"} else 0.000051
            if not math.isclose(float(row[name]), value, rel_tol=0, abs_tol=tolerance):
                failures.append(f"{method}: {name} does not match the saved topic results")

    if failures:
        print(f"Checked {checked} available topic results")
        for failure in failures[:20]:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print(f"PASS: topic coverage, summary table and metric checks for {checked} topic results")
    for method, entries in results.items():
        available = [entry for entry in entries if entry.get("available", True)]
        if available:
            avg_p = sum(entry["precision"] for entry in available) / len(available)
            avg_r = sum(entry["recall"] for entry in available) / len(available)
            avg_p10 = sum(entry["p_at_10"] for entry in available) / len(available)
            print(f"{method}: n={len(available)} precision={avg_p:.4f} recall={avg_r:.4f} P at 10={avg_p10:.4f}")

    gold = {entry["qid"]: entry for entry in results["goldilocks"]}
    one_shot = {
        entry["qid"]: entry
        for entry in results["one_shot_llm_baseline"]
        if entry.get("available", False)
    }
    paired_gold = [gold[qid] for qid in one_shot if qid in gold]
    paired_model = [one_shot[qid] for qid in one_shot if qid in gold]
    if paired_gold:
        print(f"Paired comparison topics: {len(paired_gold)}")
        for name in ("precision", "recall", "p_at_10", "hits", "in_band"):
            gold_mean = sum(item[name] for item in paired_gold) / len(paired_gold)
            model_mean = sum(item[name] for item in paired_model) / len(paired_model)
            print(f"Paired {name}: Goldilocks={gold_mean:.4f}, one shot Groq={model_mean:.4f}")

    unavailable = [
        entry for entry in results["one_shot_llm_baseline"]
        if not entry.get("available", False)
    ]
    print(f"One shot Groq unavailable topics: {len(unavailable)}")
    error_counts = {}
    for entry in unavailable:
        error = entry.get("error", "No saved error")
        error_counts[error] = error_counts.get(error, 0) + 1
    for error, count in sorted(error_counts.items()):
        print(f"Unavailable reason count {count}: {error}")


if __name__ == "__main__":
    main()
