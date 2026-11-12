"""
significance.py
Paired two-sided randomization (permutation) tests on per-topic nDCG@10 and P@10.

For a pair of runs, the per-topic differences d_i = a_i - b_i are computed. Under the null hypothesis that
the two runs are exchangeable, the sign of every d_i is random. We flip each sign with probability 1/2,
10,000 times, and count how often the absolute mean difference is at least the observed one.
p = (count + 1) / (permutations + 1). Standard library only.

Usage: python -m evaluation.significance
"""

import json
import os
import random
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_PATH = os.path.join("evaluation", "results.json")
OUTPUT_PATH = os.path.join("evaluation", "significance.json")
PAIRS = [("R1", "R2"), ("R1", "R3"), ("R1", "R4"), ("R4", "R5"), ("R0", "R4")]
METRICS = ["ndcg_at_10", "p_at_10"]
PERMUTATIONS = 10000
SEED = 42
TIE_TOLERANCE = 1e-12
ALPHA = 0.05


def randomization_p_value(differences, rng, permutations=PERMUTATIONS):
    """Two-sided paired randomization test on the mean of the differences."""
    observed = abs(mean(differences))
    count = 0
    for _ in range(permutations):
        flipped = sum(d if rng.random() < 0.5 else -d for d in differences)
        if abs(flipped / len(differences)) >= observed - TIE_TOLERANCE:
            count += 1
    return (count + 1) / (permutations + 1)


def compare(results, run_a, run_b, field, metric, rng):
    a = results["runs"][run_a]["fields"][field]["per_topic"]
    b = results["runs"][run_b]["fields"][field]["per_topic"]
    differences = [a[topic][metric] - b[topic][metric] for topic in a]
    return {
        "run_a": run_a, "run_b": run_b, "field": field, "metric": metric,
        "mean_a": mean(a[t][metric] for t in a), "mean_b": mean(b[t][metric] for t in b),
        "mean_difference": mean(differences),
        "p_value": randomization_p_value(differences, rng),
        "topics_a_wins": sum(d > TIE_TOLERANCE for d in differences),
        "topics_b_wins": sum(d < -TIE_TOLERANCE for d in differences),
        "topics_tied": sum(abs(d) <= TIE_TOLERANCE for d in differences),
    }


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    rng = random.Random(SEED)
    fields = list(results["runs"]["R1"]["fields"])
    tests = [compare(results, a, b, field, metric, rng) for field in fields for a, b in PAIRS for metric in METRICS]
    threshold = ALPHA / len(tests)
    for test in tests:
        test["significant_after_bonferroni"] = test["p_value"] < threshold
    output = {
        "method": f"paired two-sided randomization test, {PERMUTATIONS} sign flips, random.Random({SEED})",
        "tests_run": len(tests), "alpha": ALPHA, "bonferroni_threshold": threshold, "tests": tests,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"{len(tests)} tests; Bonferroni threshold {ALPHA}/{len(tests)} = {threshold:.5f}")
    for t in tests:
        print(f"{t['field']:15} {t['metric']:11} {t['run_a']} vs {t['run_b']}: diff {t['mean_difference']:+.4f}  p={t['p_value']:.4f}  "
              f"{t['run_a']} wins {t['topics_a_wins']}, {t['run_b']} wins {t['topics_b_wins']}, tied {t['topics_tied']}"
              f"{'  *' if t['significant_after_bonferroni'] else ''}")


if __name__ == "__main__":
    main()
