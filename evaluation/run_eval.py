"""
run_eval.py
Evaluates the Goldilocks search system on TREC-COVID topics.

Metrics computed:
  - Fraction of queries landing in the target band (20 to 200 hits)
  - Boolean set precision and recall (using qrels)
  - P@10 after tf-idf cosine ranking
  - Average LLM calls used per query

Baselines compared:
  1. Goldilocks controller (with or without LLM)
  2. One-shot LLM Boolean query (no controller refinement)
  3. AND of all keywords (simple baseline)
  4. OR of all keywords ranked by tf-idf (recall-oriented baseline)

Outputs: tables (printed + saved as TSV) and charts (saved as PNG).
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.search import SearchIndex
from engine.boolean_parser import parse, collect_terms, ParseError
from engine.ranker import rank_documents
from engine.controller import run_controller, load_config, keyword_terms
from engine.gemini_client import propose_boolean_query


def load_eval_data():
    """Loads cached queries and qrels from data/cache/."""
    with open("data/cache/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open("data/cache/qrels.json", "r", encoding="utf-8") as f:
        qrels = json.load(f)
    return queries, qrels


def compute_precision_recall(retrieved_ext_ids, relevant_ext_ids):
    """
    Computes precision and recall given sets of retrieved and relevant doc IDs.
    Both inputs should be sets of external (TREC) document IDs.
    """
    if not retrieved_ext_ids:
        return 0.0, 0.0
    tp = len(retrieved_ext_ids & relevant_ext_ids)
    precision = tp / len(retrieved_ext_ids) if retrieved_ext_ids else 0
    recall = tp / len(relevant_ext_ids) if relevant_ext_ids else 0
    return precision, recall


def compute_p_at_k(ranked_ext_ids, relevant_ext_ids, k=10):
    """
    Precision at K: fraction of the top K ranked results that are relevant.
    """
    top_k = ranked_ext_ids[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ext_ids)
    return relevant_count / len(top_k)


def baseline_and_keywords(query_text, index, top_k=10):
    """
    Baseline 1: AND all keywords from the query.
    Returns (doc_ids_set, ranked_results, query_str)
    """
    stems = [term for term in keyword_terms(query_text) if len(term) > 2]
    # filter to stems that exist in the index
    valid_stems = [s for s in stems if index.get_df(s) > 0]
    if not valid_stems:
        return set(), [], "EMPTY"

    query_str = " AND ".join(valid_stems)
    try:
        ast = parse(query_str)
        result_docs = index.execute_boolean(ast)
        query_terms = list(collect_terms(ast))
        ranked = rank_documents(result_docs, query_terms, index, top_k=top_k)
        return result_docs, ranked, query_str
    except ParseError:
        return set(), [], query_str


def baseline_or_keywords(query_text, index, top_k=10):
    """
    Baseline 2: OR all keywords from the query, rank by tf-idf.
    Returns (doc_ids_set, ranked_results, query_str)
    """
    stems = [term for term in keyword_terms(query_text) if len(term) > 2]
    valid_stems = [s for s in stems if index.get_df(s) > 0]
    if not valid_stems:
        return set(), [], "EMPTY"

    query_str = " OR ".join(valid_stems)
    try:
        ast = parse(query_str)
        result_docs = index.execute_boolean(ast)
        query_terms = list(collect_terms(ast))
        ranked = rank_documents(result_docs, query_terms, index, top_k=top_k)
        return result_docs, ranked, query_str
    except ParseError:
        return set(), [], query_str


def baseline_one_shot_llm(query_text, index, top_k=10):
    """Run one Gemini Boolean proposal without controller refinement."""
    try:
        query_str = propose_boolean_query(query_text)
        ast = parse(query_str)
    except RuntimeError as error:
        return {"available": False, "error": str(error)}
    except ParseError:
        return {"available": False, "error": "Gemini returned an invalid Boolean query."}

    docs = index.execute_boolean(ast)
    terms = list(collect_terms(ast))
    ranked = rank_documents(docs, terms, index, top_k=top_k)
    return {"available": True, "query": query_str, "docs": docs, "ranked": ranked}


def run_evaluation():
    """
    Runs the full evaluation on all TREC-COVID topics.
    """
    print("=" * 70)
    print("PHASE 3: Evaluation on TREC-COVID Topics")
    print("=" * 70)

    config = load_config()
    target_low = config["target_band_low"]
    target_high = config["target_band_high"]
    top_k = config["top_k"]

    # load index and evaluation data
    index = SearchIndex(max_cached_shards=4)
    queries, qrels = load_eval_data()

    print(f"Index: {index.num_docs} docs, {len(index.dictionary)} terms")
    print(f"Queries: {len(queries)}, Qrels: {len(qrels)} topics")
    print(f"Target band: [{target_low}, {target_high}]")
    print()

    # Always allow cache lookup. On a cache miss, the client uses the API key
    # if present and the controller falls back to keywords if it is absent.
    use_llm = True
    if os.environ.get("GEMINI_API_KEY"):
        print("[mode] Gemini cache and API enabled")
    else:
        print("[mode] Cached Gemini responses when available, keyword fallback otherwise")

    # results storage
    results = {
        "goldilocks": [],
        "one_shot_llm_baseline": [],
        "and_baseline": [],
        "or_baseline": [],
    }

    # track metrics
    total_queries = 0

    for qid, query_text in queries.items():
        # skip queries without qrels (cannot evaluate)
        if qid not in qrels:
            continue

        total_queries += 1
        print(f"\n--- Query {qid}: {query_text[:60]}... ---")

        # get relevant docs for this query (qrels with score > 0)
        relevant = {did for did, score in qrels[qid].items() if score > 0}

        # Reuse the proposal cache in Goldilocks after measuring one shot output.
        llm_result = baseline_one_shot_llm(query_text, index, top_k)
        if llm_result["available"]:
            llm_ext = {index.get_ext_id(doc_id) for doc_id in llm_result["docs"]}
            llm_ranked_ext = [index.get_ext_id(doc_id) for _, doc_id in llm_result["ranked"]]
            llm_p, llm_r = compute_precision_recall(llm_ext, relevant)
            results["one_shot_llm_baseline"].append({
                "qid": qid, "available": True, "hits": len(llm_result["docs"]),
                "in_band": target_low <= len(llm_result["docs"]) <= target_high,
                "precision": llm_p, "recall": llm_r,
                "p_at_10": compute_p_at_k(llm_ranked_ext, relevant, k=10),
                "query": llm_result["query"],
            })
            query_use_llm = True
            print(f"  [one shot LLM] hits={len(llm_result['docs'])}")
        else:
            results["one_shot_llm_baseline"].append({
                "qid": qid, "available": False, "error": llm_result["error"]
            })
            query_use_llm = False
            print(f"  [one shot LLM] unavailable: {llm_result['error']}")

        # ---- Goldilocks controller ----
        try:
            ranked_gc, ctrl_log, final_query = run_controller(
                query_text, index, config, use_llm=query_use_llm
            )
            final_ast = parse(final_query)
            final_doc_ids = index.execute_boolean(final_ast)
            gc_docs = {index.get_ext_id(doc_id) for doc_id in final_doc_ids}
            gc_ranked_ext = []
            for score, doc_id in ranked_gc:
                ext_id = index.get_ext_id(doc_id)
                gc_ranked_ext.append(ext_id)

            gc_actual = len(final_doc_ids)

            gc_precision, gc_recall = compute_precision_recall(gc_docs, relevant)
            gc_p10 = compute_p_at_k(gc_ranked_ext, relevant, k=10)
            gc_in_band = target_low <= gc_actual <= target_high

            results["goldilocks"].append({
                "qid": qid,
                "hits": gc_actual,
                "in_band": gc_in_band,
                "precision": gc_precision,
                "recall": gc_recall,
                "p_at_10": gc_p10,
                "llm_calls": ctrl_log.llm_calls,
                "iterations": len(ctrl_log.iterations),
                "final_query": final_query,
            })
            print(f"  [goldilocks] hits={gc_actual}, in_band={gc_in_band}, P@10={gc_p10:.3f}, llm_calls={ctrl_log.llm_calls}")
        except Exception as e:
            print(f"  [goldilocks] ERROR: {e}")
            results["goldilocks"].append({
                "qid": qid, "hits": 0, "in_band": False,
                "precision": 0, "recall": 0, "p_at_10": 0,
                "llm_calls": 0, "iterations": 0, "final_query": "ERROR",
            })

        # ---- AND baseline ----
        and_docs, and_ranked, and_query = baseline_and_keywords(query_text, index, top_k)
        and_ext = {index.get_ext_id(d) for d in and_docs}
        and_ranked_ext = [index.get_ext_id(d) for _, d in and_ranked]
        and_p, and_r = compute_precision_recall(and_ext, relevant)
        and_p10 = compute_p_at_k(and_ranked_ext, relevant, k=10)
        and_in_band = target_low <= len(and_docs) <= target_high

        results["and_baseline"].append({
            "qid": qid,
            "hits": len(and_docs),
            "in_band": and_in_band,
            "precision": and_p,
            "recall": and_r,
            "p_at_10": and_p10,
        })
        print(f"  [AND base] hits={len(and_docs)}, in_band={and_in_band}, P@10={and_p10:.3f}")

        # ---- OR baseline ----
        or_docs, or_ranked, or_query = baseline_or_keywords(query_text, index, top_k)
        or_ext = {index.get_ext_id(d) for d in or_docs}
        or_ranked_ext = [index.get_ext_id(d) for _, d in or_ranked]
        or_p, or_r = compute_precision_recall(or_ext, relevant)
        or_p10 = compute_p_at_k(or_ranked_ext, relevant, k=10)
        or_in_band = target_low <= len(or_docs) <= target_high

        results["or_baseline"].append({
            "qid": qid,
            "hits": len(or_docs),
            "in_band": or_in_band,
            "precision": or_p,
            "recall": or_r,
            "p_at_10": or_p10,
        })
        print(f"  [OR  base] hits={len(or_docs)}, in_band={or_in_band}, P@10={or_p10:.3f}")

    # ---- Aggregate and print results ----
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    for method in ["goldilocks", "one_shot_llm_baseline", "and_baseline", "or_baseline"]:
        entries = [entry for entry in results[method] if entry.get("available", True)]
        if not entries:
            if results[method]:
                print(f"\n{method.upper()}: no successful Gemini responses")
            continue

        n = len(entries)
        in_band = sum(1 for e in entries if e["in_band"])
        avg_hits = sum(e["hits"] for e in entries) / n
        avg_p = sum(e["precision"] for e in entries) / n
        avg_r = sum(e["recall"] for e in entries) / n
        avg_p10 = sum(e["p_at_10"] for e in entries) / n

        print(f"\n{method.upper()}:")
        print(f"  Queries evaluated: {n}")
        print(f"  In target band:    {in_band}/{n} ({in_band * 100 / n:.1f}%)")
        print(f"  Avg hits:          {avg_hits:.1f}")
        print(f"  Avg precision:     {avg_p:.4f}")
        print(f"  Avg recall:        {avg_r:.4f}")
        print(f"  Avg P@10:          {avg_p10:.4f}")

        if method == "goldilocks":
            avg_llm = sum(e.get("llm_calls", 0) for e in entries) / n
            avg_iter = sum(e.get("iterations", 0) for e in entries) / n
            print(f"  Avg LLM calls:     {avg_llm:.2f}")
            print(f"  Avg iterations:    {avg_iter:.2f}")

    # ---- Save results ----
    os.makedirs("evaluation", exist_ok=True)
    results_path = os.path.join("evaluation", "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    # ---- Generate charts ----
    try:
        _generate_charts(results, config)
    except Exception as e:
        print(f"[warning] Chart generation failed: {e}")

    # ---- Save summary table ----
    _save_summary_table(results, config)

    return results


def _save_summary_table(results, config):
    """Saves a summary comparison table as TSV and printed text."""
    try:
        from tabulate import tabulate as tab_fn
    except ImportError:
        print("[warning] tabulate not installed, skipping table export")
        return

    headers = ["Method", "In Band %", "Avg Hits", "Avg Precision", "Avg Recall", "Avg P@10"]
    rows = []

    for method in ["goldilocks", "one_shot_llm_baseline", "and_baseline", "or_baseline"]:
        entries = [entry for entry in results[method] if entry.get("available", True)]
        if not entries:
            continue
        n = len(entries)
        in_band_pct = sum(1 for e in entries if e["in_band"]) * 100 / n
        avg_hits = sum(e["hits"] for e in entries) / n
        avg_p = sum(e["precision"] for e in entries) / n
        avg_r = sum(e["recall"] for e in entries) / n
        avg_p10 = sum(e["p_at_10"] for e in entries) / n
        rows.append([method, f"{in_band_pct:.1f}", f"{avg_hits:.1f}", f"{avg_p:.4f}", f"{avg_r:.4f}", f"{avg_p10:.4f}"])

    table_str = tab_fn(rows, headers=headers, tablefmt="grid")
    print("\n" + table_str)

    # save as TSV
    tsv_path = os.path.join("evaluation", "summary.tsv")
    with open(tsv_path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")
    print(f"Summary table saved to {tsv_path}")


def _generate_charts(results, config):
    """Generates comparison charts and saves as PNG."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    target_low = config["target_band_low"]
    target_high = config["target_band_high"]

    # Chart 1: Hit count distribution per method
    methods = ["goldilocks", "one_shot_llm_baseline", "and_baseline", "or_baseline"]
    titles = ["Goldilocks Controller", "One Shot LLM", "AND Baseline", "OR Baseline"]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))

    for ax, method, title in zip(axes, methods, titles):
        hits = [e["hits"] for e in results[method]]
        if hits:
            ax.hist(hits, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        else:
            ax.text(0.5, 0.5, "No successful Gemini responses", ha="center", va="center", transform=ax.transAxes)
        ax.axvline(x=target_low, color="green", linestyle="--", label=f"Target low ({target_low})")
        ax.axvline(x=target_high, color="red", linestyle="--", label=f"Target high ({target_high})")
        ax.set_title(title)
        ax.set_xlabel("Hit Count")
        ax.set_ylabel("Number of Queries")
        ax.legend(fontsize=8)

    plt.tight_layout()
    chart_path = os.path.join("evaluation", "hit_distribution.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")

    # Chart 2: P@10 comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    chart_methods = []
    method_names = []
    p10_values = []
    for method in methods:
        entries = [entry for entry in results[method] if entry.get("available", True)]
        if entries:
            chart_methods.append(method)
            method_names.append(titles[methods.index(method)])
            p10_values.append(sum(e["p_at_10"] for e in entries) / len(entries))

    colors = {"goldilocks": "#2196F3", "one_shot_llm_baseline": "#9C27B0",
              "and_baseline": "#FF9800", "or_baseline": "#4CAF50"}
    bars = ax.bar(method_names, p10_values, color=[colors[method] for method in chart_methods], edgecolor="white")
    ax.set_ylabel("Average P@10")
    ax.set_title("Precision at 10 Comparison")
    ax.set_ylim(0, max(p10_values) * 1.3 if p10_values and max(p10_values) > 0 else 1)
    for bar, val in zip(bars, p10_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join("evaluation", "p10_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    run_evaluation()
