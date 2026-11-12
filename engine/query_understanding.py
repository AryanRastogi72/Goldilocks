"""
query_understanding.py
Turns a question into ordered concepts. A concept is one idea of the query:
a group of stems that are OR-ed together, with an idf computed on the union of their postings.
"""

import json
import math
import os
import re

from pipeline.tokenizer import load_query_stopwords, stem_term, tokenize_no_stem

LEXICON_PATH = os.path.join(os.path.dirname(__file__), "..", "resources", "lexicon.json")
QUOTED_RE = re.compile(r'"([^"]+)"')


def load_lexicon(path=LEXICON_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_sequence(tokens, wanted, used):
    """Position of the first unused, contiguous match of wanted in tokens, or -1."""
    for start in range(len(tokens) - len(wanted) + 1):
        window = range(start, start + len(wanted))
        if tokens[start:start + len(wanted)] == wanted and not any(i in used for i in window):
            return start
    return -1


def _group_stems(group, index):
    """Stems of the single-word variants of a lexicon group that exist in the index."""
    stems = []
    for variant in group["variants"]:
        tokens = tokenize_no_stem(variant)
        if len(tokens) == 1:
            stem = stem_term(tokens[0])
            if index.get_df(stem) > 0 and stem not in stems:
                stems.append(stem)
    return stems


def union_df(stems, index):
    """Number of documents containing at least one of the stems."""
    return len({doc_id for stem in stems for doc_id, _ in index.get_postings(stem)})


def make_concept(label, stems, index, required, source, facet=None):
    df = union_df(stems, index)
    concept = {
        "label": label,
        "stems": stems,
        "df": df,
        "idf": math.log10(index.num_docs / df),
        "required": required,
        "source": source,
        "eliminated": False,
    }
    if facet:
        concept["facet"] = facet
    return concept


def _required_concepts(text, index):
    """Every stem inside double quotes becomes its own required concept."""
    concepts = []
    for quoted in QUOTED_RE.findall(text):
        for token in tokenize_no_stem(quoted):
            stem = stem_term(token)
            if index.get_df(stem) > 0 and all(stem not in c["stems"] for c in concepts):
                concepts.append(make_concept(token, [stem], index, True, "query"))
    return concepts


def _lexicon_concepts(tokens, index, lexicon):
    """Matches lexicon variants (longest first) on the raw tokens. Returns (concepts, used positions)."""
    variants = sorted(
        ((tokenize_no_stem(v), group) for group in lexicon for v in group["variants"]),
        key=lambda pair: -len(pair[0]),
    )
    used, matched_ids, concepts = set(), set(), []
    for wanted, group in variants:
        if group["id"] in matched_ids:
            continue
        start = _find_sequence(tokens, wanted, used)
        stems = _group_stems(group, index)
        if start >= 0 and stems:
            used.update(range(start, start + len(wanted)))
            matched_ids.add(group["id"])
            concepts.append(make_concept(group["id"], stems, index, False, "lexicon", group.get("facet")))
    return concepts, used


def _word_concepts(tokens, used, index):
    stopwords = load_query_stopwords()
    concepts = []
    for position, token in enumerate(tokens):
        stem = stem_term(token)
        if position in used or token in stopwords or index.get_df(stem) == 0:
            continue
        if all(stem not in c["stems"] for c in concepts):
            concepts.append(make_concept(token, [stem], index, False, "query"))
    return concepts


def understand(text, index, config):
    """
    Returns the concepts of a query, highest idf first. Concepts whose df / N is above
    config["max_df_fraction"] are marked "eliminated" (left out of the first query),
    unless that would eliminate every concept: then the highest-idf one stays.
    """
    required = _required_concepts(text, index)
    remaining_tokens = tokenize_no_stem(QUOTED_RE.sub(" ", text))
    lexicon_concepts, used = _lexicon_concepts(remaining_tokens, index, load_lexicon())
    return mark_elimination(required + lexicon_concepts + _word_concepts(remaining_tokens, used, index), index, config)


def mark_elimination(concepts, index, config):
    """Flags too-common concepts as eliminated (keeping at least one) and sorts by idf, highest first."""
    for concept in concepts:
        too_common = concept["df"] / index.num_docs > config["max_df_fraction"]
        concept["eliminated"] = too_common and not (concept["required"] or concept.get("excluded"))
    concepts.sort(key=lambda c: -c["idf"])
    if concepts and all(c["eliminated"] for c in concepts if not c.get("excluded")):
        next(c for c in concepts if not c.get("excluded"))["eliminated"] = False
    return concepts
