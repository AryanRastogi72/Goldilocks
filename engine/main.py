"""
main.py
CLI entry point for EvidenceLens.
Usage: python -m engine.main "your search query"
       python -m engine.main --session     (one turn per line on stdin; '#2' picks clarification choice 2)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.search import SearchIndex
from engine.controller import run_controller, load_config
from engine.conversation import apply_clarification, handle_turn

SESSION_ID = "cli"
SESSION_TOP = 5


def print_usage():
    print("EvidenceLens: offline explainable search over TREC-COVID")
    print("Usage: python -m engine.main <query>")
    print("       python -m engine.main --session   (turns on stdin, '#<id>' picks a clarification)")
    print("Example: python -m engine.main 'coronavirus treatment options'")
    print('Put words in double quotes to make them required, e.g. "remdesivir" children')


def print_choices(choices):
    if not choices:
        return
    print("\nDid you mean one of these? (pick with #<id> in --session mode)")
    for choice in choices:
        print(f"  #{choice['id']}  {' / '.join(choice['label_terms'])}  ({choice['size']} papers)")


def print_trace(steps):
    for row in steps:
        print(f"  {row['step']:>2}. {row['decision']:<22} hits={row['actual_hits']:<7} source={row['source']:<10} {row['query'][:100]}")


def print_turn(result, index):
    print(f"\nI searched for: {result['rewritten_query']}")
    print("Rules fired: " + (", ".join(f"{r['rule']} ({r['detail']})" for r in result["rules_fired"]) or "none"))
    print_trace(result["trace"])
    print(f"Top {SESSION_TOP}:")
    for rank, item in enumerate(result["results"][:SESSION_TOP], 1):
        print(f"  {rank}. [{item['ext_id']}] {item['title'][:80]}")
    print_choices(result["clarifications"])


def run_session(index, config):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(f"\n> {line}")
        if line.startswith("#") and line[1:].isdigit():
            result = apply_clarification(SESSION_ID, int(line[1:]), index, config)
        else:
            result = handle_turn(SESSION_ID, line, index, config)
        print_turn(result, index)


def run_single_query(query, index, config):
    print(f"Index loaded: {index.num_docs} docs, {len(index.dictionary)} terms")
    print(f"Query: {query}")
    print(f"Target band: [{config['target_band_low']}, {config['target_band_high']}]")

    ranked, log, final_query = run_controller(query, index, config, allow_clarify=True)

    print("\nCONCEPTS")
    log.print_concepts()
    print("\nTRACE")
    log.print_trace()
    print(f"\nFINAL QUERY: {final_query}")
    print(f"STEPS: {len(log.steps)}")
    print_choices(log.clarifications)
    print(f"\nTOP {config['top_k']} RESULTS")
    for rank, (score, doc_id) in enumerate(ranked, 1):
        print(f"  {rank}. [{index.get_ext_id(doc_id)}] score={score:.4f}")
        print(f"     {index.get_doc_title(doc_id)[:80]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if not args:
        print_usage()
        sys.exit(0)
    if not os.path.exists("index_out/dictionary.json"):
        print("Error: Index not built yet.")
        print("Run: python -m pipeline.build_index")
        sys.exit(1)

    index = SearchIndex()
    config = load_config()
    if args == ["--session"]:
        run_session(index, config)
    else:
        run_single_query(" ".join(args), index, config)


if __name__ == "__main__":
    main()
