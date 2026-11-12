"""
ranker.py
Ranks a set of documents by tf-idf cosine similarity to a query.
Uses a heap-based top-K selection to avoid sorting all documents.
"""

import math
import heapq
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.tokenizer import stem_term


def compute_query_vector(query_terms, index):
    """
    Computes the tf-idf vector for a query.
    Query tf weighting: 1 + log10(tf) for each query term.
    idf: log10(N / df).

    Returns a dict: term -> weight (tf_weight * idf).
    """
    from collections import Counter

    # count how many times each term appears in the query
    tf_counter = Counter(query_terms)
    query_vec = {}

    for term, tf in tf_counter.items():
        idf = index.get_idf(term)
        if idf == 0:
            continue  # term not in vocabulary, skip
        tf_weight = 1 + math.log10(tf) if tf > 0 else 0
        query_vec[term] = tf_weight * idf

    return query_vec


def rank_documents(doc_ids, query_terms, index, top_k=10):
    """
    Ranks documents by tf-idf cosine similarity.

    For each document in doc_ids, compute the dot product of the
    document's tf-idf vector with the query's tf-idf vector, then
    divide by the document's L2 norm (precomputed) and the query's
    L2 norm.

    Uses a min-heap of size top_k to efficiently find the top-K
    documents without sorting all of them. This is O(n * log(k))
    instead of O(n * log(n)) for a full sort.

    Returns: list of (score, doc_id) tuples, highest score first.
    """
    query_vec = compute_query_vector(query_terms, index)

    if not query_vec:
        # no valid query terms, return documents in arbitrary deterministic order
        return [(0.0, doc_id) for doc_id in sorted(list(doc_ids))[:top_k]]

    # compute query norm
    query_norm = math.sqrt(sum(w ** 2 for w in query_vec.values()))
    if query_norm == 0:
        return [(0.0, doc_id) for doc_id in sorted(list(doc_ids))[:top_k]]

    # for each query term, fetch the postings and build a lookup
    # doc_id -> accumulated dot product contribution
    # only look at docs in our candidate set (doc_ids)
    doc_id_set = set(doc_ids)
    doc_scores = {}  # doc_id -> dot product

    for term, q_weight in query_vec.items():
        idf = index.get_idf(term)
        postings = index.get_postings(term)

        for doc_id, tf in postings:
            if doc_id not in doc_id_set:
                continue  # skip docs not in our Boolean result set

            # document tf weight: 1 + log10(tf)
            doc_tf_weight = (1 + math.log10(tf)) if tf > 0 else 0
            # contribution to dot product: query_weight * doc_weight
            # doc_weight is tf_weight * idf (same idf as query)
            doc_weight = doc_tf_weight * idf
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += q_weight * doc_weight

    # normalize by document norm and query norm -> cosine similarity
    scored = []
    for doc_id, dot in doc_scores.items():
        doc_norm = index.get_doc_norm(doc_id)
        if doc_norm == 0:
            continue
        cosine = dot / (doc_norm * query_norm)
        scored.append((cosine, doc_id))

    # add docs with zero score (in the Boolean set but no query term overlap)
    for doc_id in doc_id_set:
        if doc_id not in doc_scores:
            scored.append((0.0, doc_id))

    # heap-based top-K: use a min-heap of size k
    # heapq.nlargest is efficient for this: O(n * log(k))
    top = heapq.nlargest(top_k, scored, key=lambda x: x[0])

    return top


def bm25_idf(index, term):
    """Lucene idf: ln(1 + (N - df + 0.5) / (df + 0.5)), always positive."""
    df = index.get_df(term)
    return math.log(1 + (index.num_docs - df + 0.5) / (df + 0.5))


def rank_bm25(query_terms, index, top_k=10, doc_ids=None, k1=1.2, b=0.75, title_boost=0.0, term_weights=None):
    """
    Okapi BM25 with an optional title-zone boost.

    score(d) = sum over distinct query terms t of
               idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_dl))
    final(d) = score(d) + title_boost * sum of idf(t) for query terms in the title of d

    term_weights (optional dict term -> weight, default 1) scales a term's idf, for expansion terms.
    doc_ids=None scores every document that contains a query term.
    Otherwise only the given documents are scored.
    Returns: list of (score, doc_id), highest score first, ties by smaller doc_id.
    """
    allowed = None if doc_ids is None else set(doc_ids)
    avg_dl = index.get_avg_doc_length()
    scores = {}

    term_weights = term_weights or {}
    for term in set(query_terms):
        idf = bm25_idf(index, term) * term_weights.get(term, 1.0)
        for doc_id, tf in index.get_postings(term):
            if allowed is not None and doc_id not in allowed:
                continue
            length_norm = 1 - b + b * index.get_doc_length(doc_id) / avg_dl
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / (tf + k1 * length_norm)
        if title_boost:
            for doc_id, _ in index.get_title_postings(term):
                if allowed is None or doc_id in allowed:
                    scores[doc_id] = scores.get(doc_id, 0.0) + title_boost * idf

    best = heapq.nsmallest(top_k, ((-score, doc_id) for doc_id, score in scores.items()))
    return [(-negative_score, doc_id) for negative_score, doc_id in best]


if __name__ == "__main__":
    # quick test
    from engine.search import SearchIndex
    if os.path.exists("index_out/dictionary.json"):
        idx = SearchIndex()
        # get postings for "covid"
        term = stem_term("covid")
        postings = idx.get_postings(term)
        doc_ids = set(d for d, tf in postings[:100])  # take first 100
        query_terms = [stem_term("covid"), stem_term("treatment")]
        results = rank_documents(doc_ids, query_terms, idx, top_k=5)
        for score, doc_id in results:
            title = idx.get_doc_title(doc_id)
            print(f"  score={score:.4f}  doc={doc_id}  title={title[:60]}")
    else:
        print("Index not built yet.")
