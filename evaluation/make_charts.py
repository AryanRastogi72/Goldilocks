"""
make_charts.py
Draws the report charts as PNG (300 dpi, 3.4 inches wide, no titles) and saves the plotted numbers as JSON.
Reads evaluation/results.json, controller_runs*.json and index_out/evidence.json.

Usage: python -m evaluation.make_charts
"""

import json
import math
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CHARTS_DIR = os.path.join("evaluation", "charts")
WIDTH_INCHES = 3.4
DPI = 300
# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILION, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#7F7F7F"
FIELD_LABELS = {"metadata.query": "keyword query", "text": "question text"}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(figure, name, numbers):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    figure.savefig(os.path.join(CHARTS_DIR, f"{name}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    with open(os.path.join(CHARTS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2)


def chart_metrics_per_run(results):
    runs = list(results["runs"])
    fields = list(results["runs"]["R1"]["fields"])
    figure, axes = plt.subplots(len(fields), 1, figsize=(WIDTH_INCHES, 4.2), sharex=True)
    numbers = {}
    for axis, field in zip(axes, fields):
        ndcg = [results["runs"][r]["fields"][field]["mean"]["ndcg_at_10"] for r in runs]
        p10 = [results["runs"][r]["fields"][field]["mean"]["p_at_10"] for r in runs]
        positions = range(len(runs))
        axis.bar([p - 0.2 for p in positions], ndcg, width=0.4, color=BLUE, label="nDCG@10")
        axis.bar([p + 0.2 for p in positions], p10, width=0.4, color=ORANGE, label="P@10")
        axis.set_ylabel(f"score ({FIELD_LABELS[field]})")
        axis.set_ylim(0, 1)
        numbers[field] = {"runs": runs, "ndcg_at_10": ndcg, "p_at_10": p10}
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xticks(range(len(runs)))
    axes[-1].set_xticklabels(runs)
    axes[-1].set_xlabel("run")
    save(figure, "1_metrics_per_run", numbers)


def chart_topic_differences(results):
    field = "metadata.query"
    r4 = results["runs"]["R4"]["fields"][field]["per_topic"]
    r1 = results["runs"]["R1"]["fields"][field]["per_topic"]
    differences = sorted(((r4[t]["ndcg_at_10"] - r1[t]["ndcg_at_10"], t) for t in r4), reverse=True)
    figure, axis = plt.subplots(figsize=(WIDTH_INCHES, 2.4))
    colours = [BLUE if d >= 0 else VERMILION for d, _ in differences]
    axis.bar(range(len(differences)), [d for d, _ in differences], color=colours, width=0.8)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_xlabel("topics, sorted by difference")
    axis.set_ylabel("nDCG@10 difference (R4 − R1)")
    save(figure, "2_topic_differences_r4_minus_r1", {"field": field, "topics": [t for _, t in differences], "differences": [d for d, _ in differences]})


def chart_hit_counts(results):
    low, high = results["band"]
    fields = list(results["runs"]["R4"]["fields"])
    figure, axes = plt.subplots(len(fields), 1, figsize=(WIDTH_INCHES, 3.6), sharex=True)
    numbers = {"band": [low, high]}
    edges = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000]
    for axis, field in zip(axes, fields):
        hits = [max(r["hits"], 1) for r in results["runs"]["R4"]["fields"][field]["per_topic"].values()]
        axis.axvspan(low, high, color=GREEN, alpha=0.25)
        axis.hist(hits, bins=edges, color=BLUE)
        axis.set_xscale("log")
        axis.set_ylabel(f"topics ({FIELD_LABELS[field]})")
        numbers[field] = {"hits": hits}
    axes[-1].set_xlabel("R4 final hit count (log scale; shaded = band)")
    save(figure, "3_r4_hit_counts", numbers)


def chart_evidence_classes():
    data = load(os.path.join("index_out", "evidence.json"))
    counts = Counter(label[0] for label in data["labels"])
    names = [data["classes"][i] for i, _ in counts.most_common()]
    values = [counts[data["classes"].index(n)] for n in names]
    figure, axis = plt.subplots(figsize=(WIDTH_INCHES, 2.6))
    axis.barh(range(len(names)), values, color=SKY)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels([n.replace("_", " ") for n in names])
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("documents (log scale)")
    save(figure, "4_evidence_classes", {"classes": names, "documents": values, "total": sum(values)})


def chart_controller_versions():
    old, new = load(os.path.join("evaluation", "controller_runs_v2_0.json")), load(os.path.join("evaluation", "controller_runs.json"))
    fields = list(new)
    metrics = [("in_band_rate", "in-band rate"), ("mean_ndcg_at_10", "nDCG@10"), ("mean_p_at_10", "P@10")]
    figure, axes = plt.subplots(len(fields), 1, figsize=(WIDTH_INCHES, 4.0), sharex=True)
    numbers = {}
    for axis, field in zip(axes, fields):
        before = [old[field]["summary"][key] for key, _ in metrics]
        after = [new[field]["summary"][key] for key, _ in metrics]
        positions = range(len(metrics))
        axis.bar([p - 0.2 for p in positions], before, width=0.4, color=GREY, label="v2.0")
        axis.bar([p + 0.2 for p in positions], after, width=0.4, color=BLUE, label="v2.1")
        axis.set_ylim(0, 1)
        axis.set_ylabel(f"value ({FIELD_LABELS[field]})")
        numbers[field] = {"metrics": [label for _, label in metrics], "v2_0": before, "v2_1": after}
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xticks(range(len(metrics)))
    axes[-1].set_xticklabels([label for _, label in metrics])
    save(figure, "5_controller_v2_0_vs_v2_1", numbers)


def main():
    results = load(os.path.join("evaluation", "results.json"))
    chart_metrics_per_run(results)
    chart_topic_differences(results)
    chart_hit_counts(results)
    chart_evidence_classes()
    chart_controller_versions()
    print("Saved 5 charts (PNG + JSON) in", CHARTS_DIR)


if __name__ == "__main__":
    main()
