"""
export_evidence_sample.py
Exports a blind labelling sample for the evidence classifier.

15 documents per predicted class, drawn with random.Random(42) among documents whose abstract has at least
200 characters. The rows are shuffled, so the order does not reveal the predicted class.
The labelling file has an empty human_label column and no prediction; predictions and the sampling counts
needed by the scorer go to separate files. Never fills human_label.

Usage: python -m evaluation.export_evidence_sample
"""

import csv
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.evidence import CLASSES, classify

DB_PATH = os.path.join("index_out", "abstracts.sqlite")
SAMPLE_PATH = os.path.join("evaluation", "evidence_label_sample.csv")
PREDICTIONS_PATH = os.path.join("evaluation", "evidence_label_predictions.csv")
COUNTS_PATH = os.path.join("evaluation", "evidence_sampling_counts.json")
PER_CLASS = 15
MIN_ABSTRACT_CHARACTERS = 200
SEED = 42


def documents_by_predicted_class():
    by_class = defaultdict(list)
    connection = sqlite3.connect(DB_PATH)
    query = "SELECT doc_id, ext_id, title, abstract FROM docs WHERE length(abstract) >= ? ORDER BY doc_id"
    for doc_id, ext_id, title, abstract in connection.execute(query, (MIN_ABSTRACT_CHARACTERS,)):
        class_id, rule_id, _ = classify(title, abstract)
        by_class[class_id].append((doc_id, ext_id, title, abstract, rule_id))
    connection.close()
    return by_class


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    by_class = documents_by_predicted_class()
    rng = random.Random(SEED)
    sample = [(class_id, doc) for class_id in CLASSES for doc in rng.sample(by_class[class_id], PER_CLASS)]
    rng.shuffle(sample)

    write_csv(
        SAMPLE_PATH,
        ["doc_id", "ext_id", "title", "abstract", "human_label"],
        [[doc[0], doc[1], doc[2], doc[3], ""] for _, doc in sample],
    )
    write_csv(
        PREDICTIONS_PATH,
        ["doc_id", "predicted_label", "rule_id"],
        [[doc[0], class_id, doc[4]] for class_id, doc in sample],
    )
    counts = {class_id: {"population": len(by_class[class_id]), "sampled": PER_CLASS} for class_id in CLASSES}
    with open(COUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"min_abstract_characters": MIN_ABSTRACT_CHARACTERS, "seed": SEED, "classes": counts}, f, indent=2)
    print(f"Wrote {len(sample)} rows to {SAMPLE_PATH} (blind), {PREDICTIONS_PATH} and {COUNTS_PATH}")


if __name__ == "__main__":
    main()
