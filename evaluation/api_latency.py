"""
api_latency.py
Sends every topic (both fields) to POST /api/search with each ranker, in fresh sessions, through TestClient,
and reports the latency of the server-side timings. Saves evaluation/api_latency.json.

Usage: python -m evaluation.api_latency
"""

import json
import math
import os
import sys
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from api.main import app
from evaluation.quick_rank_eval import FIELDS, load_topics

OUTPUT_PATH = os.path.join("evaluation", "api_latency.json")
RANKERS = ["agent", "bm25"]
SLOW_THRESHOLD_MS = 3000


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def summarize(rows):
    totals = [r["total_ms"] for r in rows]
    return {
        "requests": len(rows), "mean_ms": mean(totals), "median_ms": median(totals), "p95_ms": percentile(totals, 0.95),
        "worst_ms": max(totals), "mean_controller_ms": mean(r["controller_ms"] for r in rows),
        "mean_explain_ms": mean(r["explain_ms"] for r in rows),
        "explain_share": sum(r["explain_ms"] for r in rows) / sum(totals),
    }


def main():
    topics = load_topics()
    rows = {}
    with TestClient(app) as client:
        client.post("/api/search", json={"text": "masks", "ranker": "bm25"})
        for ranker in RANKERS:
            for field_name, getter in FIELDS.items():
                collected = []
                for topic in topics:
                    response = client.post("/api/search", json={"text": getter(topic)[:300], "ranker": ranker, "session_id": f"lat_{ranker}_{field_name.replace('.', '_')}_{topic['_id']}"})
                    response.raise_for_status()
                    body = response.json()
                    collected.append(dict(body["timings"], topic=topic["_id"], text=getter(topic)))
                rows[(ranker, field_name)] = collected
    output = {"note": "warm-up request sent first; fresh session per request; TestClient, so no network time", "groups": {}}
    print(f"{'ranker':8}{'field':16}{'mean':>8}{'median':>8}{'p95':>8}{'worst':>8}{'explain share':>15}")
    for (ranker, field_name), collected in rows.items():
        summary = summarize(collected)
        slowest = max(collected, key=lambda r: r["total_ms"])
        summary["slowest_topic"] = {k: slowest[k] for k in ("topic", "text", "total_ms", "controller_ms", "explain_ms")}
        output["groups"][f"{ranker}/{field_name}"] = summary
        print(f"{ranker:8}{field_name:16}{summary['mean_ms']:8.0f}{summary['median_ms']:8.0f}{summary['p95_ms']:8.0f}{summary['worst_ms']:8.0f}{summary['explain_share']:15.1%}")
    everything = [r for collected in rows.values() for r in collected]
    output["all"] = summarize(everything)
    worst = max(everything, key=lambda r: r["total_ms"])
    output["worst_request"] = {k: worst[k] for k in ("topic", "text", "total_ms", "controller_ms", "explain_ms")}
    if worst["total_ms"] > SLOW_THRESHOLD_MS:
        part = "controller" if worst["controller_ms"] >= worst["explain_ms"] else "explanations"
        output["slowest_part_above_3s"] = part
        print(f"Worst request {worst['total_ms']:.0f} ms is above {SLOW_THRESHOLD_MS} ms; slowest part: {part}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"All {output['all']['requests']} requests: mean {output['all']['mean_ms']:.0f} ms, median {output['all']['median_ms']:.0f} ms, "
          f"p95 {output['all']['p95_ms']:.0f} ms, worst {output['all']['worst_ms']:.0f} ms, explanations {output['all']['explain_share']:.1%} of the time")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
