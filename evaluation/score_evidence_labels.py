"""
score_evidence_labels.py
Scores the evidence classifier against the human labels.

The sample was stratified by predicted class, so:
  - precision of a class is unbiased: it is the share of sampled documents predicted as that class that
    humans labelled the same;
  - recall, F1 and accuracy are re-weighted: every sampled document of predicted class c stands for
    weight_c = population_c / labelled_c documents of the corpus (abstract of at least 200 characters).

Usage: python -m evaluation.score_evidence_labels [sample.csv predictions.csv counts.json]
"""

import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.evidence import CLASSES

SAMPLE_PATH = os.path.join("evaluation", "evidence_label_sample.csv")
PREDICTIONS_PATH = os.path.join("evaluation", "evidence_label_predictions.csv")
COUNTS_PATH = os.path.join("evaluation", "evidence_sampling_counts.json")


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def labelled_pairs(sample_path, predictions_path):
    """Joins the two files on doc_id for rows with a human label. Raises ValueError on an unknown label."""
    predictions = {row["doc_id"]: row["predicted_label"] for row in read_rows(predictions_path)}
    pairs = []
    for row in read_rows(sample_path):
        label = row["human_label"].strip()
        if not label:
            continue
        if label not in CLASSES:
            raise ValueError(f"row doc_id={row['doc_id']}: unknown label '{label}'; expected one of {CLASSES}")
        pairs.append((predictions[row["doc_id"]], label))
    return pairs


def confusion(pairs, weights):
    """Weighted counts: matrix[(predicted, human)] = sum of the weight of the predicted class."""
    matrix = Counter()
    for predicted, human in pairs:
        matrix[(predicted, human)] += weights[predicted]
    return matrix


def metrics(matrix):
    """Per-class precision, recall, F1 and overall accuracy from a (possibly weighted) confusion matrix."""
    per_class = {}
    for class_id in CLASSES:
        predicted_total = sum(v for (p, _), v in matrix.items() if p == class_id)
        human_total = sum(v for (_, h), v in matrix.items() if h == class_id)
        correct = matrix[(class_id, class_id)]
        precision = correct / predicted_total if predicted_total else None
        recall = correct / human_total if human_total else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        per_class[class_id] = {"precision": precision, "recall": recall, "f1": f1}
    total = sum(matrix.values())
    accuracy = sum(matrix[(c, c)] for c in CLASSES) / total if total else None
    return per_class, accuracy


def format_value(value):
    return "  n/a " if value is None else f"{value:6.3f}"


def print_table(title, per_class, accuracy):
    print(f"\n{title}")
    print(f"  {'class':<34}{'precision':>10}{'recall':>8}{'F1':>8}")
    for class_id, row in per_class.items():
        print(f"  {class_id:<34}{format_value(row['precision']):>10}{format_value(row['recall']):>8}{format_value(row['f1']):>8}")
    print(f"  overall accuracy: {format_value(accuracy).strip()}")


def score(sample_path=SAMPLE_PATH, predictions_path=PREDICTIONS_PATH, counts_path=COUNTS_PATH):
    pairs = labelled_pairs(sample_path, predictions_path)
    with open(counts_path, "r", encoding="utf-8") as f:
        population = {c: v["population"] for c, v in json.load(f)["classes"].items()}
    labelled = Counter(predicted for predicted, _ in pairs)
    weights = {c: population[c] / labelled[c] for c in labelled}
    weighted = metrics(confusion(pairs, weights))
    unweighted = metrics(confusion(pairs, Counter({c: 1 for c in CLASSES})))
    return len(pairs), weighted, unweighted


def main():
    paths = sys.argv[1:4] if len(sys.argv) >= 4 else []
    count, weighted, unweighted = score(*paths)
    if count == 0:
        print("No labelled rows yet. Fill human_label in the sample file and run again.")
        return
    print(f"{count} labelled rows")
    print_table("Re-weighted by inverse sampling probability (precision unbiased; recall, F1, accuracy estimated for the corpus)", *weighted)
    print_table("Unweighted sample numbers (the sample is balanced by predicted class, so recall and accuracy are not corpus estimates)", *unweighted)


if __name__ == "__main__":
    main()
