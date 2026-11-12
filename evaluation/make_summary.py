"""
make_summary.py
Writes evaluation/summary.md from results.json, significance.json, beir_reference.json and the qrels.
Usage: python -m evaluation.make_summary
"""

import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.quick_rank_eval import FIELDS, load_qrels, load_topics

SUMMARY_MD = os.path.join("evaluation", "summary.md")
FIELD_LABELS = {"metadata.query": "keyword query (`metadata.query`)", "text": "question text (`text`)"}
EXAMPLES_SHOWN = 5


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_table(results):
    lines = ["| run | field | P@10 | nDCG@10 | MRR@10 | MAP@100 | Recall@100 | Recall@1000 | latency s |", "|---|---|---|---|---|---|---|---|---|"]
    for run_name, run in results["runs"].items():
        for field_name, field in run["fields"].items():
            m = field["mean"]
            lines.append(
                f"| {run_name} | {field_name} | {m['p_at_10']:.4f} | {m['ndcg_at_10']:.4f} | {m['mrr_at_10']:.4f} | "
                f"{m['map_at_100']:.4f} | {m['recall_at_100']:.4f} | {m['recall_at_1000']:.4f} | {field['mean_latency_s']:.3f} |"
            )
    return lines


def set_table(results):
    lines = ["| run | field | mean hits | set precision | set recall | set F1 | in band |", "|---|---|---|---|---|---|---|"]
    for run_name, run in results["runs"].items():
        if run["kind"] != "set":
            continue
        for field_name, field in run["fields"].items():
            m = field["mean"]
            lines.append(
                f"| {run_name} | {field_name} | {m['hits']:.1f} | {m['set_precision']:.4f} | {m['set_recall']:.4f} | "
                f"{m['set_f1']:.4f} | {m['in_band']:.0%} |"
            )
    return lines


def recall_note(results, qrels):
    relevant = [sum(1 for g in topic.values() if g > 0) for topic in qrels.values()]
    r4 = {f: results["runs"]["R4"]["fields"][f]["mean"]["recall_at_1000"] for f in results["runs"]["R4"]["fields"]}
    r1 = {f: results["runs"]["R1"]["fields"][f]["mean"]["recall_at_1000"] for f in results["runs"]["R1"]["fields"]}
    return (
        f"Note on Recall@1000: R4 and R5 return one accepted set, and the band caps that set at about {results['band'][1]} documents "
        f"(mean {results['runs']['R4']['fields']['text']['mean']['hits']:.0f} hits on the question text), while the 50 topics average "
        f"{mean(relevant):.1f} judged-relevant documents (minimum {min(relevant)}, maximum {max(relevant)}; computed from the qrels). "
        f"Their Recall@1000 can therefore never reach that of a ranking run: R4 {r4['metadata.query']:.4f} / {r4['text']:.4f} against R1 {r1['metadata.query']:.4f} / {r1['text']:.4f} "
        "(keyword query / question text). The agent trades recall for a reading-sized result set."
    )


def beir_section(results, beir):
    ours = results["runs"]["R1"]["fields"]["text"]["mean"]["ndcg_at_10"]
    return [
        "## BEIR comparison (question text field, nDCG@10)",
        "",
        f"BEIR paper (Thakur et al., arXiv 2104.08663, version {beir['paper_version_read']}), Table {beir['table']}: BM25 on TREC-COVID nDCG@10 = {beir['bm25_trec_covid_ndcg10']}. "
        f"Our R1 (BM25, question text field) = {ours:.4f}; difference {ours - beir['bm25_trec_covid_ndcg10']:+.4f}.",
        "",
        f"The gap is {abs(ours - beir['bm25_trec_covid_ndcg10']):.4f} nDCG@10 points, which is within what small implementation differences can produce; "
        "with different k1, b, tokenizer and field handling an exact match would be a coincidence, so we do not claim that we reproduce BEIR's pipeline.",
        "",
        "Differences that could contribute (none was tested one by one):",
        f"- BEIR's BM25 is Anserini with Lucene defaults k1 = {beir['beir_k1']}, b = {beir['beir_b']}; ours uses k1 = 1.2, b = 0.75 and a Lucene-style idf.",
        "- BEIR indexes title and text as separate fields; we index one stream of title plus abstract tokens with a single length.",
        "- Tokenizer and stemming differ (our regex tokenizer with Porter stemming and no stopword removal while indexing; query stopwords are removed on our side).",
        "- Lucene may store document length in a lossy encoded norm; we use exact token counts (this detail comes from general knowledge, not from the BEIR paper, and was not checked).",
        "- The BEIR corpus version and judgement handling may differ from the files we use (our run uses the same 171,332 documents and 50 topics).",
        "",
    ]


AI_SECTION_TITLE = "Evidence classifier: agreement with reference labels, not human-validated accuracy"


def score_table(per_class, accuracy):
    lines = ["| class | precision | recall | F1 |", "|---|---|---|---|"]
    for class_id, row in per_class.items():
        cells = [f"{row[k]:.3f}" if row[k] is not None else "n/a" for k in ("precision", "recall", "f1")]
        lines.append(f"| {class_id} | " + " | ".join(cells) + " |")
    return lines + [f"| overall | accuracy {accuracy:.3f} | | |"]


def evidence_section():
    human_path = os.path.join("evaluation", "evidence_scores.json")
    ai_path = os.path.join("evaluation", "evidence_scores_ai.json")
    lines = ["## Evidence classifier accuracy (human labels)", ""]
    if os.path.exists(human_path):
        scores = load(human_path)
        lines += [f"Re-weighted accuracy {scores['weighted_accuracy']:.3f} on {scores['labelled_rows']} human-labelled rows (documents with abstracts only). Details in `evaluation/evidence_scores.json`.", ""]
    else:
        lines += ["TODO(metric): the blind sample `evaluation/evidence_label_sample.csv` has not been labelled by a human yet; "
                  "after labelling, `python -m evaluation.score_evidence_labels` saves `evaluation/evidence_scores.json`.", ""]
    if os.path.exists(ai_path):
        ai = load(ai_path)
        lines += [f"## {AI_SECTION_TITLE}", "",
                  f"Labels were made blind on {ai['labelled_rows']} sampled documents with abstracts. "
                  "Details in `evaluation/evidence_scores_ai.json`.", "",
                  "Re-weighted by inverse sampling probability (precision unbiased; recall, F1 and accuracy estimated for the corpus):", ""]
        lines += score_table(ai["weighted_per_class"], ai["weighted_accuracy"])
        lines += ["", "Unweighted sample numbers (the sample is balanced by predicted class, so recall and accuracy are not corpus estimates):", ""]
        lines += score_table(ai["unweighted_per_class"], ai["unweighted_accuracy"]) + [""]
    return lines


def significance_lines(significance):
    lines = [f"Bonferroni threshold for {significance['tests_run']} tests: {significance['alpha']} / {significance['tests_run']} = {significance['bonferroni_threshold']:.4f}.", ""]
    lines += ["| field | metric | pair | mean difference (first minus second) | p | first wins | second wins | tied | significant after Bonferroni |", "|---|---|---|---|---|---|---|---|---|"]
    for t in significance["tests"]:
        lines.append(
            f"| {t['field']} | {t['metric']} | {t['run_a']} vs {t['run_b']} | {t['mean_difference']:+.4f} | {t['p_value']:.4f} | "
            f"{t['topics_a_wins']} | {t['topics_b_wins']} | {t['topics_tied']} | {'yes' if t['significant_after_bonferroni'] else 'no'} |"
        )
    return lines


def comparison_sentence(test):
    higher, lower = (test["run_a"], test["run_b"]) if test["mean_difference"] >= 0 else (test["run_b"], test["run_a"])
    verdict = "significant after Bonferroni correction" if test["significant_after_bonferroni"] else (
        "not significant at 0.05" if test["p_value"] >= 0.05 else "significant at 0.05 but not after Bonferroni correction")
    return (f"- {test['run_a']} vs {test['run_b']}, {test['metric']}: {higher} is higher by {abs(test['mean_difference']):.4f} "
            f"(p = {test['p_value']:.4f}, {test['run_a']} wins {test['topics_a_wins']}, {test['run_b']} wins {test['topics_b_wins']}, tied {test['topics_tied']}); {verdict}.")


def top_differences(results, field, topic_text):
    """The topics where R4 beats R1 most and loses most on nDCG@10, as lists of dicts."""
    r1 = results["runs"]["R1"]["fields"][field]["per_topic"]
    r4 = results["runs"]["R4"]["fields"][field]["per_topic"]
    ranked = sorted(((r4[t]["ndcg_at_10"] - r1[t]["ndcg_at_10"], t) for t in r4), reverse=True)

    def row(difference, topic):
        return {
            "topic": topic, "text": topic_text[field][topic], "r1_ndcg_at_10": r1[topic]["ndcg_at_10"],
            "r4_ndcg_at_10": r4[topic]["ndcg_at_10"], "difference": difference,
            "final_query": r4[topic]["final_query"], "hits": r4[topic]["hits"],
        }

    return [row(d, t) for d, t in ranked[:EXAMPLES_SHOWN]], [row(d, t) for d, t in ranked[::-1][:EXAMPLES_SHOWN]]


def wins_and_losses(results, significance, topic_text):
    lines = ["## Where EvidenceLens wins and loses", "", "R4 is the EvidenceLens agent (controller v2.1) and R1 is plain BM25 on the full collection."]
    for field in FIELDS:
        wins, losses = top_differences(results, field, topic_text)
        lines += ["", f"### {FIELD_LABELS[field]}", "", "Comparisons:"]
        lines += [comparison_sentence(t) for t in significance["tests"] if t["field"] == field]
        for title, rows in (("Topics where R4 beats R1 most (nDCG@10)", wins), ("Topics where R4 loses to R1 most (nDCG@10)", losses)):
            lines += ["", f"{title}:", "", "| topic | text | R1 | R4 | difference | R4 final Boolean query | R4 hits |", "|---|---|---|---|---|---|---|"]
            for r in rows:
                query = r["final_query"].replace("|", "/")
                lines.append(f"| {r['topic']} | {r['text']} | {r['r1_ndcg_at_10']:.3f} | {r['r4_ndcg_at_10']:.3f} | "
                             f"{r['difference']:+.3f} | `{query}` | {r['hits']} |")
    return lines


def main():
    results, significance = load(os.path.join("evaluation", "results.json")), load(os.path.join("evaluation", "significance.json"))
    beir, qrels = load(os.path.join("evaluation", "beir_reference.json")), load_qrels()
    topics = load_topics()
    topic_text = {field: {t["_id"]: getter(t) for t in topics} for field, getter in FIELDS.items()}
    lines = ["# Evaluation summary", "", results["note"] + ".", "",
             f"All runs use depth {results['depth']}, 50 topics, relevant = grade above 0, nDCG with linear gain. Run files: `evaluation/runs/*.trec`; recomputed by `python -m evaluation.validate_results`.", "",
             "## Ranking metrics (mean over 50 topics)", ""]
    lines += metric_table(results) + ["", recall_note(results, qrels), "", "## Set metrics for set runs (R0, R4, R5)", ""]
    lines += set_table(results) + ["", "## Significance (paired randomization tests)", ""]
    lines += significance_lines(significance) + [""]
    lines += beir_section(results, beir) + evidence_section() + wins_and_losses(results, significance, topic_text)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
