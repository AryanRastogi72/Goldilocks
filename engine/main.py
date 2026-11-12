"""
main.py
CLI entry point for EvidenceLens.
Usage: python -m engine.main "your search query"
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.search import SearchIndex
from engine.controller import run_controller, load_config


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("EvidenceLens: offline explainable search over TREC-COVID")
        print("Usage: python -m engine.main <query>")
        print("Example: python -m engine.main 'coronavirus treatment options'")
        print('Put words in double quotes to make them required, e.g. "remdesivir" children')
        sys.exit(0)

    query = " ".join(sys.argv[1:])

    if not os.path.exists("index_out/dictionary.json"):
        print("Error: Index not built yet.")
        print("Run: python -m pipeline.build_index")
        sys.exit(1)

    index = SearchIndex()
    config = load_config()
    print(f"Index loaded: {index.num_docs} docs, {len(index.dictionary)} terms")
    print(f"Query: {query}")
    print(f"Target band: [{config['target_band_low']}, {config['target_band_high']}]")

    ranked, log, final_query = run_controller(query, index, config)

    print("\nCONCEPTS")
    log.print_concepts()
    print("\nTRACE")
    log.print_trace()
    print(f"\nFINAL QUERY: {final_query}")
    print(f"STEPS: {len(log.steps)}")
    print(f"\nTOP {config['top_k']} RESULTS")
    for rank, (score, doc_id) in enumerate(ranked, 1):
        print(f"  {rank}. [{index.get_ext_id(doc_id)}] score={score:.4f}")
        print(f"     {index.get_doc_title(doc_id)[:80]}")


if __name__ == "__main__":
    main()
