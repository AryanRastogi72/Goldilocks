"""Print reproducible relevance coverage facts from saved evaluation files."""

import gzip
import json

from engine.search import SearchIndex


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    with open("data/cache/qrels.json", "r", encoding="utf-8") as file:
        qrels = json.load(file)
    with open("evaluation/results.json", "r", encoding="utf-8") as file:
        results = json.load(file)
    with gzip.open("evaluation/retrieved_ids.json.gz", "rt", encoding="utf-8") as file:
        retrieved = json.load(file)

    index = SearchIndex(max_cached_shards=1)
    indexed_ids = {item["ext_id"] for item in index.doc_meta.values()}
    relevant_ids = {
        doc_id for topic in qrels.values()
        for doc_id, grade in topic.items() if grade > 0
    }
    missing_ids = relevant_ids - indexed_ids
    print(f"Relevant judged documents: {len(relevant_ids)}")
    print(f"Relevant judged documents absent from index: {len(missing_ids)}")

    for method, entries in results.items():
        available = [entry for entry in entries if entry.get("available", True)]
        recalls = [entry["recall"] for entry in available]
        zero_recall = sum(value == 0 for value in recalls)
        missed = 0
        retrieved_relevant = 0
        for entry in available:
            relevant = {
                doc_id for doc_id, grade in qrels[entry["qid"]].items() if grade > 0
            }
            found = set(retrieved[method][entry["qid"]]) & relevant
            missed += len(relevant - found)
            retrieved_relevant += len(found)
        print(
            f"{method}: topics={len(available)}, mean recall={mean(recalls):.4f}, "
            f"zero recall topics={zero_recall}, missed relevant judgments={missed}, "
            f"retrieved relevant judgments={retrieved_relevant}"
        )


if __name__ == "__main__":
    main()
