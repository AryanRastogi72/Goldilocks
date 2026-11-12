"""The evaluation data served by GET /api/eval, built from the saved evaluation files."""

import glob
import json
import os

from evaluation.make_summary import top_differences
from evaluation.quick_rank_eval import FIELDS, load_topics

EVAL_DIR = "evaluation"
AI_NOTE = "agreement with reference labels, not human-validated accuracy"


def load(name):
    with open(os.path.join(EVAL_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def exists(name):
    return os.path.exists(os.path.join(EVAL_DIR, name))


def run_tables(results):
    """Per-run, per-field means without the per-topic records."""
    return {
        run_name: {
            "description": run["description"], "kind": run["kind"],
            "fields": {f: {"mean": v["mean"], "mean_latency_s": v["mean_latency_s"]} for f, v in run["fields"].items()},
        }
        for run_name, run in results["runs"].items()
    }


def wins_and_losses(results):
    topics = load_topics()
    topic_text = {field: {t["_id"]: getter(t) for t in topics} for field, getter in FIELDS.items()}
    output = {}
    for field in FIELDS:
        wins, losses = top_differences(results, field, topic_text)
        output[field] = {"r4_beats_r1_most": wins, "r4_loses_to_r1_most": losses}
    return output


def chart_data():
    charts = {}
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "charts", "*.json"))):
        charts[os.path.splitext(os.path.basename(path))[0]] = json.load(open(path, encoding="utf-8"))
    return charts


def evidence_block():
    human = load("evidence_scores.json") if exists("evidence_scores.json") else {"status": "TODO(metric)"}
    ai = {"source": "ai", "note": AI_NOTE, "status": "not available"}
    if exists("evidence_scores_ai.json"):
        ai = dict(load("evidence_scores_ai.json"), source="ai", note=AI_NOTE)
    return {"human": human, "ai": ai}


def build_eval_payload():
    results = load("results.json")
    beir = load("beir_reference.json")
    beir["our_r1_text_ndcg_at_10"] = results["runs"]["R1"]["fields"]["text"]["mean"]["ndcg_at_10"]
    return {
        "note": results["note"],
        "depth": results["depth"],
        "band": results["band"],
        "runs": run_tables(results),
        "significance": load("significance.json"),
        "beir": beir,
        "wins_and_losses": wins_and_losses(results),
        "charts": chart_data(),
        "controller_versions": {
            "v2_0": {f: v["summary"] for f, v in load("controller_runs_v2_0.json").items()},
            "v2_1": {f: v["summary"] for f, v in load("controller_runs.json").items()},
        },
        "evidence_classifier": evidence_block(),
    }
