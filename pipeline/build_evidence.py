"""
build_evidence.py
Classifies every document with the rules in engine/evidence.py.

Usage: python -m pipeline.build_evidence
Writes index_out/evidence.json: {"classes": [...], "rules": [...], "labels": [[class_idx, rule_idx], ...]}
with one label per doc_id.
"""

import json
import os
import sqlite3
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.evidence import CLASSES, NO_MATCH, RULES, classify

INDEX_DIR = "index_out"
TOP_RULES_SHOWN = 10


def classify_all(db_path):
    rule_ids = [rule["id"] for rule in RULES] + [NO_MATCH]
    labels, missing_abstract = [], 0
    connection = sqlite3.connect(db_path)
    for title, abstract in connection.execute("SELECT title, abstract FROM docs ORDER BY doc_id"):
        class_id, rule_id, _ = classify(title, abstract)
        labels.append([CLASSES.index(class_id), rule_ids.index(rule_id)])
        missing_abstract += not (abstract or "").strip()
    connection.close()
    return labels, rule_ids, missing_abstract


def main():
    start = time.time()
    labels, rule_ids, missing_abstract = classify_all(os.path.join(INDEX_DIR, "abstracts.sqlite"))
    output = {"classes": CLASSES, "rules": rule_ids, "labels": labels}
    with open(os.path.join(INDEX_DIR, "evidence.json"), "w") as f:
        json.dump(output, f, separators=(",", ":"))

    total = len(labels)
    print(f"Classified {total} documents in {time.time() - start:.1f}s")
    print("\nClass distribution")
    for class_idx, count in sorted(Counter(label[0] for label in labels).items(), key=lambda item: -item[1]):
        print(f"  {CLASSES[class_idx]:<34}{count:>8}  {count / total:6.1%}")
    print(f"\nTop {TOP_RULES_SHOWN} rules")
    for rule_idx, count in Counter(label[1] for label in labels).most_common(TOP_RULES_SHOWN):
        print(f"  {rule_ids[rule_idx]:<12}{count:>8}")
    print(f"\nDocuments with no abstract: {missing_abstract} ({missing_abstract / total:.1%})")
    size_mb = os.path.getsize(os.path.join(INDEX_DIR, "evidence.json")) / (1024 * 1024)
    print(f"evidence.json: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
