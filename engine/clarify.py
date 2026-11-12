"""
clarify.py
Clarifying questions by cluster pruning: leader-follower clustering of the top candidates,
labelled with the terms that distinguish each cluster from the whole candidate set.
"""

import math
import random
from collections import Counter, defaultdict

from engine.expansion import stemmed_stopwords
from engine.ranker import rank_bm25
from pipeline.tokenizer import stem_term, tokenize_no_stem

MAX_CHOICES = 4
LABEL_TERMS = 3
LABEL_MIN_DF = 5


def document_vector(title, abstract, index, stem_cache, surface_forms):
    """ltc vector of a document: (1 + log10 tf) * idf, length-normalised. Also counts surface forms per stem."""
    counts = Counter()
    for token in tokenize_no_stem(f"{title} {abstract}"):
        if token not in stem_cache:
            stem_cache[token] = stem_term(token)
        stem = stem_cache[token]
        counts[stem] += 1
        surface_forms[stem][token] += 1
    weights = {}
    for stem, tf in counts.items():
        idf = index.get_idf(stem)
        if idf > 0:
            weights[stem] = (1 + math.log10(tf)) * idf
    norm = math.sqrt(sum(w * w for w in weights.values()))
    return {stem: w / norm for stem, w in weights.items()} if norm else {}


def cosine(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(w * right.get(stem, 0.0) for stem, w in left.items())


def leader_follower(vectors, seed):
    """
    Picks ceil(sqrt(n)) random leaders, then puts every vector with the leader of highest cosine.
    Returns a list of clusters, each a list of positions in vectors, largest first.
    """
    leader_count = math.ceil(math.sqrt(len(vectors)))
    leaders = random.Random(seed).sample(range(len(vectors)), leader_count)
    clusters = [[] for _ in leaders]
    for position, vector in enumerate(vectors):
        best = max(range(len(leaders)), key=lambda i: (cosine(vector, vectors[leaders[i]]), -i))
        clusters[best].append(position)
    return sorted((c for c in clusters if c), key=len, reverse=True)


def cluster_label_stems(cluster, vectors, excluded_stems, index):
    """Stems ranked by (mean weight inside the cluster) - (mean weight over all candidates)."""
    overall = defaultdict(float)
    for vector in vectors:
        for stem, weight in vector.items():
            overall[stem] += weight
    inside = defaultdict(float)
    for position in cluster:
        for stem, weight in vectors[position].items():
            inside[stem] += weight
    contrast = {
        stem: total / len(cluster) - overall[stem] / len(vectors)
        for stem, total in inside.items()
        if stem not in excluded_stems and not stem.isdigit() and index.get_df(stem) >= LABEL_MIN_DF
    }
    return sorted(contrast, key=lambda stem: (-contrast[stem], stem))


def pick_labels(ranked_stems, taken_first_labels):
    first = next((stem for stem in ranked_stems if stem not in taken_first_labels), None)
    if first is None:
        return []
    others = [stem for stem in ranked_stems if stem != first][: LABEL_TERMS - 1]
    return [first] + others


def build_clarifications(doc_ids, concepts, index, config):
    """
    Returns up to 4 choices {"id", "label_terms", "label_stems", "size"} for the largest clusters
    (at least config["clarify_min_size"] documents) among the top candidates of doc_ids.
    """
    query_stems = sorted({stem for c in concepts if not c.get("excluded") for stem in c["stems"]})
    ranked = rank_bm25(query_stems, index, top_k=config["clarify_candidates"], doc_ids=doc_ids)
    if len(ranked) < 2 * config["clarify_min_size"]:
        return []
    stem_cache, surface_forms = {}, defaultdict(Counter)
    vectors = [document_vector(*index.get_title_and_abstract(doc_id), index, stem_cache, surface_forms) for _, doc_id in ranked]
    clusters = leader_follower(vectors, config["seed"])
    big_clusters = [c for c in clusters if len(c) >= config["clarify_min_size"]][:MAX_CHOICES]

    excluded_stems = set(query_stems) | stemmed_stopwords()
    choices, taken_first_labels = [], set()
    for cluster in big_clusters:
        label_stems = pick_labels(cluster_label_stems(cluster, vectors, excluded_stems, index), taken_first_labels)
        if not label_stems:
            continue
        taken_first_labels.add(label_stems[0])
        choices.append({
            "id": len(choices) + 1,
            "label_terms": [surface_forms[stem].most_common(1)[0][0] for stem in label_stems],
            "label_stems": label_stems,
            "size": len(cluster),
        })
    return choices if len(choices) >= 2 else []
