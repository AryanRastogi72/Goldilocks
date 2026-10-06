"""
main.py
CLI entry point for the Goldilocks search engine.
Usage: python -m engine.main "your search query"
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.search import SearchIndex
from engine.controller import run_controller, load_config


def main():
    if len(sys.argv) < 2:
        print("Goldilocks: Agentic Boolean Search Engine")
        print("Usage: python -m engine.main <query>")
        print("Example: python -m engine.main 'coronavirus treatment options'")
        print()
        print("Set GEMINI_API_KEY env var to enable LLM query proposal.")
        print("Without it, the engine uses keyword AND as the initial query.")
        sys.exit(0)

    query = " ".join(sys.argv[1:])

    # check that the index exists
    if not os.path.exists("index_out/dictionary.json"):
        print("Error: Index not built yet.")
        print("Run: python -m pipeline.build_index")
        sys.exit(1)

    print(f"Loading index...")
    index = SearchIndex()
    config = load_config()

    print(f"Index loaded: {index.num_docs} docs, {len(index.dictionary)} terms")
    print(f"Query: {query}")
    print(f"Target band: [{config['target_band_low']}, {config['target_band_high']}]")
    print()

    # Always permit cache lookup. On a cache miss, the client uses the API key
    # if present and the controller falls back to keywords if it is absent.
    use_llm = True
    if os.environ.get("GEMINI_API_KEY"):
        print("[mode] Gemini cache and API enabled")
    else:
        print("[mode] Cached Gemini responses when available, keyword fallback otherwise")

    # run the controller loop
    ranked, ctrl_log, final_query = run_controller(query, index, config, use_llm=use_llm)

    # print the iteration trace
    print("\n" + "=" * 60)
    print("ITERATION TRACE")
    print("=" * 60)
    ctrl_log.print_trace()

    # print final results
    top_k = config.get("top_k", 10)
    print("\n" + "=" * 60)
    print(f"FINAL QUERY: {final_query}")
    print(f"LLM CALLS USED: {ctrl_log.llm_calls}")
    print(f"TOTAL ITERATIONS: {len(ctrl_log.iterations)}")
    print(f"\nTOP {top_k} RESULTS:")
    print("-" * 60)
    for rank, (score, doc_id) in enumerate(ranked, 1):
        title = index.get_doc_title(doc_id)
        ext_id = index.get_ext_id(doc_id)
        print(f"  {rank}. [{ext_id}] score={score:.4f}")
        print(f"     {title[:80]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
