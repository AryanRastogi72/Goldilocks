"""
controller.py
The agent controller (v2). It builds a Boolean query from idf-ordered concepts, runs it on the
index, and tightens or relaxes the query with rules until the hit count is in the target band.

Every decision uses only index statistics (df, estimated hits, actual hits). No model API is called.
Every step is logged with the source that proposed the move:
  query, lexicon, thesaurus, prf, kgram or rule.

v2.1: the first query ANDs only the top `initial_concepts` concepts (plus required ones). Too many hits are
tightened with the user's own words first (more concepts, then the title zone) and only then with feedback terms.
"""

import json
import os

from engine.boolean_parser import ast_to_string, collect_terms
from engine.clarify import build_clarifications
from engine.expansion import prf_terms, thesaurus_neighbours
from engine.query_understanding import union_df, understand
from engine.ranker import rank_bm25

PRF_DOCS = 10
SYNONYMS_PER_MOVE = 3
WILDCARD_TERMS_PER_MOVE = 5


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


class ControllerLog:
    """Stores the concepts and the trace of every step for display and evaluation."""

    def __init__(self, concepts):
        self.concepts = concepts
        self.steps = []
        self.clarifications = []

    def add(self, query, df_table, estimated_hits, actual_hits, decision, reason, source):
        self.steps.append({
            "step": len(self.steps) + 1,
            "query": query,
            "df_table": df_table,
            "estimated_hits": estimated_hits,
            "actual_hits": actual_hits,
            "decision": decision,
            "reason": reason,
            "source": source,
        })

    def to_dict(self):
        return {"concepts": self.concepts, "steps": self.steps, "clarifications": self.clarifications}

    def print_concepts(self):
        print(f"{'concept':<24}{'stems':<44}{'idf':>6}  flags")
        for concept in self.concepts:
            flags = ", ".join(
                name for name in ("required", "eliminated", "held", "excluded") if concept.get(name)
            ) or "-"
            print(f"{concept['label']:<24}{' '.join(concept['stems']):<44}{concept['idf']:>6.2f}  {flags}")

    def print_trace(self):
        for row in self.steps:
            print(
                f"{row['step']:>2}. {row['decision']:<22} hits={row['actual_hits']:<7} "
                f"est={row['estimated_hits']:<7} source={row['source']:<10} {row['query']}"
            )
            print(f"      {row['reason']}")


# ---- query building ----

def is_active(concept):
    return not (concept["eliminated"] or concept.get("held") or concept.get("dropped") or concept.get("excluded"))


def concept_node(concept):
    terms = [("TERM", stem) for stem in concept["stems"]]
    return terms[0] if len(terms) == 1 else ("OR", terms)


def build_ast(concepts, extra_nodes):
    included = [concept_node(c) for c in concepts if is_active(c)] + extra_nodes
    excluded = [("NOT", concept_node(c)) for c in concepts if c.get("excluded")]
    children = included + excluded
    return children[0] if len(children) == 1 else ("AND", children)


def concept_docs(concept, index):
    return {doc_id for stem in concept["stems"] for doc_id, _ in index.get_postings(stem)}


def current_df(concept, index):
    return union_df(concept["stems"], index)


def active_concepts(state, include_required):
    return [c for c in state["concepts"] if is_active(c) and (include_required or not c["required"])]


def already_used(state, move, key):
    return (move, key) in state["used_moves"]


def mark_used(state, move, key):
    state["used_moves"].add((move, key))


# ---- tighten moves: each returns (decision, reason, source) or None ----

def tighten_add_concept(state, docs, index, config):
    """Adds the next unused user concept: held ones first (by idf), then eliminated common ones."""
    pending = [
        c for c in state["concepts"]
        if (c.get("held") or c["eliminated"]) and not c.get("dropped") and not c.get("excluded")
        and not already_used(state, "TIGHTEN_ADD_CONCEPT", c["label"])
    ]
    for concept in sorted(pending, key=lambda c: (c["eliminated"], -c["idf"])):
        hits = len(docs & concept_docs(concept, index))
        if hits >= config["target_band_low"]:
            mark_used(state, "TIGHTEN_ADD_CONCEPT", concept["label"])
            was_common = concept["eliminated"]
            concept["eliminated"] = False
            concept["held"] = False
            kind = "common concept" if was_common else "concept"
            return "TIGHTEN_ADD_CONCEPT", f"Added {kind} '{concept['label']}' (idf {concept['idf']:.2f}); {hits} hits remain", "rule"
    return None


def tighten_add_prf(state, docs, index, config):
    query_stems = sorted({s for c in active_concepts(state, True) for s in c["stems"]})
    for stem, weight in prf_terms(query_stems, docs, index, n_docs=PRF_DOCS, max_df_fraction=config["max_df_fraction"]):
        if already_used(state, "TIGHTEN_ADD_PRF", stem):
            continue
        hits = len(docs & {doc_id for doc_id, _ in index.get_postings(stem)})
        if hits >= config["target_band_low"]:
            mark_used(state, "TIGHTEN_ADD_PRF", stem)
            state["extra_nodes"].append(("TERM", stem))
            return "TIGHTEN_ADD_PRF", f"Added feedback term '{stem}' (Rocchio weight {weight:.2f}) from the top {PRF_DOCS} documents; {hits} hits remain", "prf"
    return None


def tighten_title_zone(state, docs, index, config):
    for concept in sorted(active_concepts(state, True), key=lambda c: -c["idf"]):
        if already_used(state, "TIGHTEN_TITLE_ZONE", concept["label"]):
            continue
        best_stem = min(concept["stems"], key=index.get_df)
        hits = len(docs & {doc_id for doc_id, _ in index.get_title_postings(best_stem)})
        if hits >= config["target_band_low"]:
            mark_used(state, "TIGHTEN_TITLE_ZONE", concept["label"])
            state["extra_nodes"].append(("TITLE", best_stem))
            return "TIGHTEN_TITLE_ZONE", f"Required '{best_stem}' (concept '{concept['label']}') in the title; {hits} hits remain", "rule"
    return None


TIGHTEN_MOVES = [tighten_add_concept, tighten_title_zone, tighten_add_prf]


# ---- relax moves ----

def relax_expand_synonyms(state, docs, index, config):
    for concept in sorted(active_concepts(state, False), key=lambda c: current_df(c, index)):
        if already_used(state, "RELAX_EXPAND_SYNONYMS", concept["label"]):
            continue
        neighbours = {}
        for stem in concept["stems"]:
            for neighbour, cosine in thesaurus_neighbours(stem, index):
                if neighbour not in concept["stems"] and index.get_df(neighbour) > 0:
                    neighbours[neighbour] = max(cosine, neighbours.get(neighbour, 0))
        added = sorted(neighbours, key=lambda n: -neighbours[n])[:SYNONYMS_PER_MOVE]
        if added:
            mark_used(state, "RELAX_EXPAND_SYNONYMS", concept["label"])
            concept["stems"] = concept["stems"] + added
            return "RELAX_EXPAND_SYNONYMS", f"Added thesaurus neighbours {added} to concept '{concept['label']}'", "thesaurus"
    return None


def wildcard_prefix(stem):
    if len(stem) >= 6:
        return stem[:-2]
    return stem[:-1] if len(stem) >= 4 else stem


def relax_wildcard(state, docs, index, config):
    for concept in sorted(active_concepts(state, False), key=lambda c: current_df(c, index)):
        if already_used(state, "RELAX_WILDCARD", concept["label"]):
            continue
        rarest_stem = min(concept["stems"], key=index.get_df)
        prefix = wildcard_prefix(rarest_stem)
        expanded = [t for t in index.expand_wildcard(prefix) if t not in concept["stems"]]
        added = sorted(expanded, key=lambda t: -index.get_df(t))[:WILDCARD_TERMS_PER_MOVE]
        if added:
            mark_used(state, "RELAX_WILDCARD", concept["label"])
            concept["stems"] = concept["stems"] + added
            return "RELAX_WILDCARD", f"Expanded '{rarest_stem}' with prefix '{prefix}*' to {added}", "kgram"
    return None


def relax_drop_concept(state, docs, index, config):
    candidates = active_concepts(state, False)
    if len(active_concepts(state, True)) < 2 or not candidates:
        return None
    concept = min(candidates, key=lambda c: current_df(c, index))
    concept["dropped"] = True
    return "RELAX_DROP_CONCEPT", f"Dropped the rarest concept '{concept['label']}' (df {current_df(concept, index)})", "rule"


RELAX_MOVES = [relax_expand_synonyms, relax_wildcard, relax_drop_concept]


# ---- main loop ----

def first_applicable_move(moves, state, docs, index, config):
    for move in moves:
        outcome = move(state, docs, index, config)
        if outcome:
            return outcome
    return None


def log_step(log, ast, index, docs, decision, reason, source):
    log.add(ast_to_string(ast), index.get_df_table(ast), index.estimate_hits(ast), len(docs), decision, reason, source)


def fresh_concepts(concepts):
    """Copies concepts and clears the per-run flags."""
    copies = [dict(c, stems=list(c["stems"])) for c in concepts]
    for concept in copies:
        concept.pop("held", None)
        concept.pop("dropped", None)
    return copies


def hold_back_extra_concepts(concepts, initial_concepts):
    """Marks concepts beyond the top `initial_concepts` (non-eliminated, non-required, by idf) as held."""
    candidates = [c for c in concepts if not (c["eliminated"] or c["required"] or c.get("excluded"))]
    for concept in sorted(candidates, key=lambda c: -c["idf"])[initial_concepts:]:
        concept["held"] = True


def log_clarify(log, ast, index, docs, concepts, config, allow_clarify, reason):
    if not allow_clarify:
        log_step(log, ast, index, docs, "CLARIFY", f"{reason}; clarification not offered", "rule")
        return
    log.clarifications = build_clarifications(docs, concepts, index, config)
    outcome = f"{len(log.clarifications)} choices offered" if log.clarifications else "no clear clusters, ranking the current set"
    log_step(log, ast, index, docs, "CLARIFY", f"{reason}; {outcome}", "rule")


def run_controller(text, index, config=None, concepts=None, allow_clarify=False):
    """
    Understands the query (or uses the given concepts), then loops: run the Boolean query, compare the
    hit count with the band, and tighten or relax. Returns (ranked, log, final_query).
    With allow_clarify the log may carry clarification choices; without it a CLARIFY step is only logged.
    """
    config = config or load_config()
    low, high = config["target_band_low"], config["target_band_high"]
    concepts = fresh_concepts(understand(text, index, config) if concepts is None else concepts)
    hold_back_extra_concepts(concepts, config["initial_concepts"])
    log = ControllerLog(concepts)
    if not any(not c.get("excluded") for c in concepts):
        log.add("", {}, 0, 0, "EXHAUSTED", "no searchable terms in the question", "rule")
        return [], log, ""
    state = {"concepts": concepts, "extra_nodes": [], "used_moves": set()}
    comparable_concepts = [c for c in concepts if not c["eliminated"] and not c.get("excluded")]

    for step in range(1, config["max_iterations"] + 1):
        ast = build_ast(concepts, state["extra_nodes"])
        docs = index.execute_boolean(ast)
        if low <= len(docs) <= high:
            log_step(log, ast, index, docs, "ACCEPT", f"Hit count {len(docs)} is in target band [{low}, {high}]", "rule")
            break
        too_many = len(docs) > high
        if too_many and step == 1 and allow_clarify and len(comparable_concepts) == 1:
            log_clarify(log, ast, index, docs, concepts, config, allow_clarify, f"one concept with {len(docs)} hits (above {high})")
            break
        if step == config["max_iterations"]:
            if too_many:
                log_clarify(log, ast, index, docs, concepts, config, allow_clarify, f"iteration limit reached with {len(docs)} hits")
            else:
                log_step(log, ast, index, docs, "EXHAUSTED", f"Iteration limit reached with {len(docs)} hits", "rule")
            break
        outcome = first_applicable_move(TIGHTEN_MOVES if too_many else RELAX_MOVES, state, docs, index, config)
        if outcome is None and too_many:
            log_clarify(log, ast, index, docs, concepts, config, allow_clarify, f"{len(docs)} hits is above {high} and no tighten move applies")
            break
        if outcome is None:
            log_step(log, ast, index, docs, "EXHAUSTED", f"{len(docs)} hits is below {low} and no relax move applies", "rule")
            break
        log_step(log, ast, index, docs, *outcome)

    ast = build_ast(concepts, state["extra_nodes"])
    docs = index.execute_boolean(ast)
    ranked = rank_final(ast, docs, concepts, index, config, log)
    return ranked, log, ast_to_string(ast)


def rank_final(ast, docs, concepts, index, config, log):
    ranking_options = dict(top_k=config["top_k"], k1=config["bm25_k1"], b=config["bm25_b"], title_boost=config["title_boost"])
    wanted = [c for c in concepts if not c.get("excluded")]
    all_stems = sorted({stem for c in wanted for stem in c["stems"]})
    if docs:
        ranking_stems = {stem for c in wanted if not c.get("dropped") for stem in c["stems"]} | collect_terms(ast)
        return rank_bm25(sorted(ranking_stems), index, doc_ids=docs, **ranking_options)
    reason = "no Boolean match; full-collection BM25 ranking shown"
    if log.steps[-1]["decision"] == "EXHAUSTED":
        log.steps[-1]["reason"] += f"; {reason}"
    else:
        log_step(log, ast, index, docs, "EXHAUSTED", reason, "rule")
    return rank_bm25(all_stems, index, **ranking_options)
