"""
controller.py
The Goldilocks controller loop: iteratively refines a Boolean query
until the result set falls within a target band (default 20 to 200 hits).

The controller uses ONLY index statistics (df, estimated hits, actual hits)
to make decisions. The LLM only proposes terms and synonyms.

Each iteration is logged with: query version, df table, estimated bound,
actual hits, decision taken, and the reason for it.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.boolean_parser import parse, ast_to_string, collect_terms, ParseError
from engine.search import SearchIndex
from engine.ranker import rank_documents
from engine.gemini_client import propose_boolean_query, suggest_synonyms, refine_query
from pipeline.tokenizer import stem_term, tokenize


def keyword_terms(text):
    """Return stemmed keyword terms, excluding Boolean operator words."""
    return [term for term in tokenize(text) if term not in {"and", "or", "not"}]


def keyword_query(text, operator="AND"):
    """Build a Boolean fallback query from normalized user keywords."""
    return f" {operator} ".join(keyword_terms(text))


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


class ControllerLog:
    """Stores the trace of each iteration for display and evaluation."""

    def __init__(self):
        self.iterations = []
        self.llm_calls = 0  # count of Gemini API calls used
        self.llm_errors = []

    def add(self, iteration_num, query_str, df_table, estimated_hits, actual_hits, decision, reason):
        self.iterations.append({
            "iteration": iteration_num,
            "query": query_str,
            "df_table": df_table,
            "estimated_hits": estimated_hits,
            "actual_hits": actual_hits,
            "decision": decision,
            "reason": reason,
        })

    def to_dict(self):
        return {
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
        }

    def print_trace(self):
        """Prints a human-readable trace of all iterations."""
        for it in self.iterations:
            print(f"\n--- Iteration {it['iteration']} ---")
            print(f"  Query: {it['query']}")
            print(f"  DF table: {it['df_table']}")
            print(f"  Estimated hits: {it['estimated_hits']}")
            print(f"  Actual hits: {it['actual_hits']}")
            print(f"  Decision: {it['decision']}")
            print(f"  Reason: {it['reason']}")


def run_controller(user_query, index, config=None, use_llm=True):
    """
    Main controller loop.

    1. Ask LLM (or use keywords) for an initial Boolean query
    2. Parse it, check df of each term, estimate hit count
    3. Execute the query, get actual hit count
    4. If hits are in the target band, stop and rank
    5. If too many hits: tighten (add a high-idf AND term, or add NOT)
    6. If too few hits: relax (drop lowest-df AND term, expand wildcards, add OR synonyms)
    7. Repeat until in band or max iterations reached

    Returns: (ranked_results, log)
    """
    if config is None:
        config = load_config()

    target_low = config.get("target_band_low", 20)
    target_high = config.get("target_band_high", 200)
    max_iter = config.get("max_iterations", 8)
    top_k = config.get("top_k", 10)

    log = ControllerLog()

    # Step 1: Get initial Boolean query
    if use_llm:
        try:
            raw_query = propose_boolean_query(user_query)
            log.llm_calls += 1
        except RuntimeError as e:
            # fallback: just AND all keywords
            print(f"[controller] LLM unavailable ({e}), using keyword fallback")
            log.llm_errors.append(str(e))
            raw_query = keyword_query(user_query)
    else:
        # no LLM: AND all keywords as the initial query
        raw_query = keyword_query(user_query)

    current_query_str = raw_query.strip()

    for iteration in range(1, max_iter + 1):
        # parse the current query
        try:
            ast = parse(current_query_str)
        except ParseError as e:
            # if parse fails, fall back to ANDing keywords
            log.add(iteration, current_query_str, {}, 0, 0, "PARSE_ERROR", str(e))
            current_query_str = keyword_query(current_query_str)
            continue

        query_str_clean = ast_to_string(ast)

        # get df table for logging
        df_table = index.get_df_table(ast)

        # estimate hit count
        estimated = index.estimate_hits(ast)

        # execute the query to get actual hit count
        result_docs = index.execute_boolean(ast)
        actual = len(result_docs)

        # decide what to do
        if target_low <= actual <= target_high:
            # in the sweet spot, stop
            decision = "ACCEPT"
            reason = f"Hit count {actual} is in target band [{target_low}, {target_high}]"
            log.add(iteration, query_str_clean, df_table, estimated, actual, decision, reason)
            break

        elif actual > target_high:
            # too many hits, need to tighten
            decision, reason, new_query = _tighten(
                ast, df_table, actual, target_high, index, user_query, log, use_llm
            )
            log.add(iteration, query_str_clean, df_table, estimated, actual, decision, reason)
            if new_query and new_query.strip() != current_query_str.strip():
                current_query_str = new_query
            else:
                break

        elif actual < target_low:
            # too few hits (or zero), need to relax
            decision, reason, new_query = _relax(
                ast, df_table, actual, target_low, index, user_query, log, use_llm
            )
            log.add(iteration, query_str_clean, df_table, estimated, actual, decision, reason)
            if new_query and new_query.strip() != current_query_str.strip():
                current_query_str = new_query
            else:
                break

        else:
            # should not happen, but just in case
            decision = "ACCEPT"
            reason = "Fallthrough"
            log.add(iteration, query_str_clean, df_table, estimated, actual, decision, reason)
            break

    # final execution and ranking
    try:
        final_ast = parse(current_query_str)
    except ParseError:
        # last resort fallback
        words = user_query.split()
        final_ast = parse(keyword_query(user_query, "OR"))

    final_docs = index.execute_boolean(final_ast)
    query_terms = list(collect_terms(final_ast))
    ranked = rank_documents(final_docs, query_terms, index, top_k=top_k)

    return ranked, log, current_query_str


def _tighten(ast, df_table, actual, target_high, index, user_query, log, use_llm):
    """
    Strategy to reduce hits when there are too many.
    Options (tried in order):
    1. Add a high-idf AND term from the user query that is not already in the query
    2. Ask LLM for a more specific term to AND
    3. Add NOT for a very common term unrelated to the query
    """
    terms_in_query = collect_terms(ast)

    # Strategy 1: Find a high-idf term from the user query not yet in the Boolean query
    user_tokens = [t for t in keyword_terms(user_query) if len(t) > 2]
    candidates = []
    for t in user_tokens:
        if t not in terms_in_query and index.get_df(t) > 0:
            candidates.append((t, index.get_idf(t)))

    if candidates:
        # pick the highest idf term (most discriminative)
        candidates.sort(key=lambda x: x[1], reverse=True)
        new_term = candidates[0][0]
        new_query = f"({ast_to_string(ast)}) AND {new_term}"
        return "TIGHTEN_ADD_AND", f"Added high-idf AND term '{new_term}' (idf={candidates[0][1]:.2f})", new_query

    # Strategy 2: Ask LLM for refinement
    if use_llm:
        try:
            feedback = f"The query returned {actual} hits, which is too many (target: at most {target_high}). Add more specific AND terms to narrow results."
            refined = refine_query(ast_to_string(ast), feedback, user_query)
            log.llm_calls += 1
            return "TIGHTEN_LLM_REFINE", f"LLM refined query to reduce from {actual} hits", refined.strip()
        except RuntimeError as e:
            log.llm_errors.append(str(e))
            pass

    # Strategy 3: If the query is a pure OR, try converting to AND
    if ast[0] == "OR":
        children = ast[1]
        new_query = " AND ".join(ast_to_string(c) for c in children)
        return "TIGHTEN_OR_TO_AND", "Converted OR to AND to reduce hits", new_query

    # no more strategies, keep current
    return "TIGHTEN_EXHAUSTED", "No more tightening strategies available", None


def _relax(ast, df_table, actual, target_low, index, user_query, log, use_llm):
    """
    Strategy to increase hits when there are too few (or zero).
    Options (tried in order):
    1. Drop the lowest-df AND term (if query is AND)
    2. Expand terms with wildcards via k-gram index
    3. Ask LLM for synonyms and add them as OR alternatives
    4. Convert AND to OR
    """
    terms_in_query = collect_terms(ast)

    # Strategy 1: Drop the lowest-df AND term
    if ast[0] == "AND" and len(ast[1]) > 2:
        children = ast[1]
        # find child with lowest estimated hits
        child_sizes = []
        for i, child in enumerate(children):
            est = index.estimate_hits(child)
            child_sizes.append((est, i))
        child_sizes.sort()
        # drop the one with smallest df (most restrictive)
        drop_idx = child_sizes[0][1]
        dropped_str = ast_to_string(children[drop_idx])
        remaining = [c for i, c in enumerate(children) if i != drop_idx]
        if len(remaining) == 1:
            new_query = ast_to_string(remaining[0])
        else:
            new_query = " AND ".join(ast_to_string(c) for c in remaining)
        return "RELAX_DROP_AND", f"Dropped lowest-df AND term '{dropped_str}'", new_query

    # Strategy 2: Wildcard expansion for the rarest term
    if terms_in_query:
        rarest = min(terms_in_query, key=lambda t: index.get_df(t))
        if len(rarest) >= 3:
            expanded = index.expand_wildcard(rarest[:3])
            if len(expanded) > 1:
                or_terms = " OR ".join(expanded[:5])  # limit to 5 expansions
                # replace the rarest term with the OR expansion
                new_query = ast_to_string(ast).replace(rarest, f"({or_terms})")
                return "RELAX_WILDCARD", f"Expanded '{rarest}' via k-gram to {len(expanded)} terms", new_query

    # Strategy 3: Ask LLM for synonyms
    if use_llm and terms_in_query:
        try:
            rarest = min(terms_in_query, key=lambda t: index.get_df(t))
            synonyms = suggest_synonyms(rarest, user_query)
            log.llm_calls += 1
            if synonyms:
                stemmed_syns = [stem_term(s) for s in synonyms]
                # keep only synonyms that exist in the index
                valid_syns = [s for s in stemmed_syns if index.get_df(s) > 0]
                if valid_syns:
                    or_part = " OR ".join([rarest] + valid_syns[:3])
                    new_query = ast_to_string(ast).replace(rarest, f"({or_part})")
                    return "RELAX_SYNONYMS", f"Added synonyms for '{rarest}': {valid_syns[:3]}", new_query
        except RuntimeError as e:
            log.llm_errors.append(str(e))
            pass

    # Strategy 4: Convert AND to OR (last resort)
    if ast[0] == "AND":
        children = ast[1]
        new_query = " OR ".join(ast_to_string(c) for c in children)
        return "RELAX_AND_TO_OR", "Converted AND to OR as last resort", new_query

    # no more strategies
    return "RELAX_EXHAUSTED", "No more relaxation strategies available", None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m engine.controller <query>")
        print("Example: python -m engine.controller 'coronavirus treatment'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    if not os.path.exists("index_out/dictionary.json"):
        print("Index not built yet. Run: python -m pipeline.build_index")
        sys.exit(1)

    index = SearchIndex()
    config = load_config()

    # try with LLM first, fall back to keyword-only
    use_llm = bool(os.environ.get("GEMINI_API_KEY"))
    ranked, ctrl_log, final_query = run_controller(query, index, config, use_llm=use_llm)

    print("\n" + "=" * 60)
    print("CONTROLLER TRACE")
    print("=" * 60)
    ctrl_log.print_trace()

    print("\n" + "=" * 60)
    print(f"FINAL QUERY: {final_query}")
    print(f"LLM CALLS: {ctrl_log.llm_calls}")
    print(f"TOP {config.get('top_k', 10)} RESULTS:")
    print("=" * 60)
    for score, doc_id in ranked:
        title = index.get_doc_title(doc_id)
        ext_id = index.get_ext_id(doc_id)
        print(f"  [{ext_id}] score={score:.4f}  {title[:70]}")
