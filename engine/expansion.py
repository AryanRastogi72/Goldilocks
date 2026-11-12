"""
expansion.py
Query expansion without any model API: word2vec thesaurus neighbours (global)
and Rocchio pseudo-relevance feedback terms (local).
"""

import json
import math
import os
from collections import Counter

from engine.ranker import rank_bm25
from pipeline.tokenizer import load_query_stopwords, stem_term, tokenize

PRF_MIN_DF = 5

_thesaurus = {}
_stopword_stems = None


def thesaurus_neighbours(stem, index):
    """Returns [[neighbour, cosine], ...] for a stem from index_out/thesaurus.json, or []."""
    if index.index_dir not in _thesaurus:
        path = os.path.join(index.index_dir, "thesaurus.json")
        with open(path, "r") as f:
            _thesaurus[index.index_dir] = json.load(f)
    return _thesaurus[index.index_dir].get(stem, [])


def _stemmed_stopwords():
    global _stopword_stems
    if _stopword_stems is None:
        _stopword_stems = {stem_term(word) for word in load_query_stopwords()}
    return _stopword_stems


def prf_terms(query_stems, doc_ids, index, n_docs=10, n_terms=10, max_df_fraction=0.10):
    """
    Rocchio centroid of the top n_docs documents (BM25 order) of doc_ids, or of the
    whole collection when doc_ids is None. A candidate term t gets the weight
    sum over those documents of (1 + log10 tf) * idf(t): the beta part of Rocchio
    with every top document assumed relevant and no negative set.

    Returns [(stem, weight)], highest first.
    """
    top_docs = rank_bm25(query_stems, index, top_k=n_docs, doc_ids=doc_ids)
    excluded = set(query_stems) | _stemmed_stopwords()
    weights = Counter()
    for _, doc_id in top_docs:
        title, abstract = index.get_title_and_abstract(doc_id)
        for term, tf in Counter(tokenize(f"{title} {abstract}")).items():
            if term in excluded or term.isdigit():
                continue
            df = index.get_df(term)
            if df < PRF_MIN_DF or df / index.num_docs > max_df_fraction:
                continue
            weights[term] += (1 + math.log10(tf)) * index.get_idf(term)
    return weights.most_common(n_terms)


def rank_bm25_prf(query_stems, index, top_k=10, alpha=1.0, beta=0.5, n_terms=10,
                  k1=1.2, b=0.75, title_boost=0.0, max_df_fraction=0.10):
    """
    Two-pass BM25. Pass 1 finds PRF terms; pass 2 re-ranks with original terms
    weighted alpha and expansion terms weighted beta * weight / max_weight.
    """
    expansion = prf_terms(query_stems, None, index, n_terms=n_terms, max_df_fraction=max_df_fraction)
    if not expansion:
        return rank_bm25(query_stems, index, top_k=top_k, k1=k1, b=b, title_boost=title_boost)
    max_weight = expansion[0][1]
    term_weights = {stem: alpha for stem in query_stems}
    term_weights.update({stem: beta * weight / max_weight for stem, weight in expansion})
    terms = list(query_stems) + [stem for stem, _ in expansion]
    return rank_bm25(terms, index, top_k=top_k, k1=k1, b=b, title_boost=title_boost, term_weights=term_weights)
