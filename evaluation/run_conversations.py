"""
run_conversations.py
Runs the scripted conversations and saves what each turn did. These are case studies, not a benchmark.
A turn written as '#<id>' picks that clarification choice from the previous turn.
Usage: python -m evaluation.run_conversations
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.controller import load_config
from engine.conversation import apply_clarification, handle_turn
from engine.search import SearchIndex

SCRIPTS_PATH = os.path.join("evaluation", "conversation_scripts.json")
RUNS_PATH = os.path.join("evaluation", "conversation_runs.json")
TOP_SHOWN = 5


def run_turn(session_id, turn, index, config):
    if turn.startswith("#"):
        return apply_clarification(session_id, int(turn[1:]), index, config)
    return handle_turn(session_id, turn, index, config)


def summarize_turn(turn, result):
    return {
        "turn": turn,
        "rewritten_query": result["rewritten_query"],
        "rules_fired": result["rules_fired"],
        "boolean_query": result["boolean_query"],
        "evidence_filter": result["evidence_filter"],
        "trace": [{k: row[k] for k in ("step", "decision", "actual_hits", "source")} for row in result["trace"]],
        "clarifications": result["clarifications"],
        "top_titles": [item["title"] for item in result["results"][:TOP_SHOWN]],
    }


def main():
    index = SearchIndex()
    config = load_config()
    with open(SCRIPTS_PATH, "r", encoding="utf-8") as f:
        scripts = json.load(f)

    sessions = []
    for script in scripts:
        print(f"\n== {script['id']}: {script['description']}")
        turns = []
        for turn in script["turns"]:
            result = run_turn(script["id"], turn, index, config)
            turns.append(summarize_turn(turn, result))
            rules = ", ".join(r["rule"] for r in result["rules_fired"]) or "none"
            print(f"  > {turn}\n    searched: {result['rewritten_query']}\n    rules: {rules}")
        sessions.append({"id": script["id"], "description": script["description"], "turns": turns})

    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        json.dump({"note": "case studies, not a benchmark", "sessions": sessions}, f, indent=2)
    print(f"\nSaved {RUNS_PATH} ({len(sessions)} sessions)")


if __name__ == "__main__":
    main()
