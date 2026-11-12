"""Ranking metrics computed the way trec_eval computes them. Grades <= 0 count as not relevant."""

import math


def compute_p_at_k(ranked_ext_ids, relevant_ext_ids, k=10):
    """Precision at K: fraction of the top K ranked results that are relevant."""
    if k <= 0:
        return 0.0
    relevant_count = sum(1 for doc_id in ranked_ext_ids[:k] if doc_id in relevant_ext_ids)
    return relevant_count / k


def compute_recall_at_k(ranked_ext_ids, relevant_ext_ids, k=100):
    """Recall at K: fraction of all relevant documents found in the top K."""
    if not relevant_ext_ids:
        return 0.0
    found = sum(1 for doc_id in ranked_ext_ids[:k] if doc_id in relevant_ext_ids)
    return found / len(relevant_ext_ids)


def compute_ndcg_at_k(ranked_ext_ids, topic_qrels, k=10):
    """nDCG with linear gain: gain = grade, discount = 1 / log2(rank + 1)."""
    if k <= 0:
        return 0.0

    def gain_at(rank, grade):
        return max(grade, 0) / math.log2(rank + 1)

    actual = sum(
        gain_at(rank, topic_qrels.get(doc_id, 0))
        for rank, doc_id in enumerate(ranked_ext_ids[:k], start=1)
    )
    ideal_grades = sorted((grade for grade in topic_qrels.values() if grade > 0), reverse=True)[:k]
    ideal = sum(gain_at(rank, grade) for rank, grade in enumerate(ideal_grades, start=1))
    return actual / ideal if ideal else 0.0


def compute_mrr_at_k(ranked_ext_ids, topic_qrels, k=10):
    """Reciprocal rank of the first relevant result within the top K."""
    for rank, doc_id in enumerate(ranked_ext_ids[:k], start=1):
        if topic_qrels.get(doc_id, 0) > 0:
            return 1 / rank
    return 0.0


def compute_map_at_k(ranked_ext_ids, relevant_ext_ids, k=100):
    """trec_eval map_cut: sum of precision at each relevant rank up to k, divided by the number of relevant documents."""
    if not relevant_ext_ids:
        return 0.0
    found, precision_sum = 0, 0.0
    for rank, doc_id in enumerate(ranked_ext_ids[:k], start=1):
        if doc_id in relevant_ext_ids:
            found += 1
            precision_sum += found / rank
    return precision_sum / len(relevant_ext_ids)


def compute_set_metrics(retrieved_ext_ids, relevant_ext_ids):
    """Precision, recall and F1 of a retrieved set (not a ranking)."""
    hits = len(retrieved_ext_ids & relevant_ext_ids)
    precision = hits / len(retrieved_ext_ids) if retrieved_ext_ids else 0.0
    recall = hits / len(relevant_ext_ids) if relevant_ext_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"set_precision": precision, "set_recall": recall, "set_f1": f1}
