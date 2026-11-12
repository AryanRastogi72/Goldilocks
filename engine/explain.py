"""
explain.py
"Why this result": which query stems matched, how much each added to the BM25 score, the evidence class
with the rule that fired, and the abstract sentence closest to the query.

Usage: python -m engine.explain <ext_id> "<query>"
"""

import bisect
import json
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.evidence import classify
from engine.query_understanding import understand
from engine.ranker import bm25_idf
from engine.search import SearchIndex
from pipeline.tokenizer import stem_term, tokenize

WORD_RE = re.compile(r"[A-Za-z0-9]+")
ABBREVIATIONS = ["e.g.", "i.e.", "et al.", "vs.", "approx.", "fig.", "no."]
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
DOT_PLACEHOLDER = "\u0001"


def term_frequency(postings, doc_id):
    """tf of a document in a postings list sorted by doc id, or 0."""
    position = bisect.bisect_left(postings, [doc_id, 0])
    return postings[position][1] if position < len(postings) and postings[position][0] == doc_id else 0


def bm25_contributions(doc_id, stems, index, config):
    """Per-stem BM25 part and title-boost part. Their sum over stems equals rank_bm25's score for this document."""
    length_norm = 1 - config["bm25_b"] + config["bm25_b"] * index.get_doc_length(doc_id) / index.get_avg_doc_length()
    k1 = config["bm25_k1"]
    parts = []
    for stem in sorted(set(stems)):
        tf = term_frequency(index.get_postings(stem), doc_id)
        if not tf:
            continue
        idf = bm25_idf(index, stem)
        in_title = term_frequency(index.get_title_postings(stem), doc_id) > 0
        parts.append({
            "stem": stem,
            "tf": tf,
            "bm25": idf * tf * (k1 + 1) / (tf + k1 * length_norm),
            "title_boost": config["title_boost"] * idf if in_title else 0.0,
        })
    return parts


def split_sentences(text):
    """Splits on sentence-ending punctuation, protecting common abbreviations such as e.g. and et al."""
    for abbreviation in ABBREVIATIONS:
        text = re.sub(re.escape(abbreviation), abbreviation.replace(".", DOT_PLACEHOLDER), text, flags=re.IGNORECASE)
    return [s.replace(DOT_PLACEHOLDER, ".").strip() for s in SENTENCE_END.split(text) if s.strip()]


def ltc_vector(text, index):
    counts = Counter(tokenize(text))
    weights = {stem: (1 + math.log10(tf)) * index.get_idf(stem) for stem, tf in counts.items() if index.get_df(stem)}
    norm = math.sqrt(sum(w * w for w in weights.values()))
    return {stem: w / norm for stem, w in weights.items()} if norm else {}


def highlight_offsets(sentence, stems):
    """Character (start, end) offsets of the words in the sentence whose stem is a query stem."""
    return [(m.start(), m.end()) for m in WORD_RE.finditer(sentence) if stem_term(m.group(0)) in stems]


def best_sentence(abstract, stems, index):
    if not abstract.strip():
        return None, "no abstract"
    query_vector = ltc_vector(" ".join(sorted(stems)), index)
    scored = []
    for sentence in split_sentences(abstract):
        vector = ltc_vector(sentence, index)
        scored.append((sum(w * query_vector.get(stem, 0.0) for stem, w in vector.items()), sentence))
    score, sentence = max(scored, key=lambda pair: pair[0])
    if score <= 0:
        return None, "no sentence contains a query term"
    return {"text": sentence, "cosine": score, "highlights": highlight_offsets(sentence, stems)}, None


def explain(doc_id, concepts, index, config, extra_stems=()):
    """
    Explanation of one result. extra_stems adds terms the controller used beyond the concepts
    (feedback and title-zone terms), so the contributions match the score shown for the result.
    """
    stems = {stem for c in concepts if not c.get("excluded") for stem in c["stems"]} | set(extra_stems)
    title, abstract = index.get_title_and_abstract(doc_id)
    parts = bm25_contributions(doc_id, stems, index, config)
    class_id, rule_id, matched = classify(title, abstract)
    sentence, missing_reason = best_sentence(abstract or "", stems, index)
    return {
        "doc_id": doc_id,
        "ext_id": index.get_ext_id(doc_id),
        "title": title,
        "matched_stems": [
            {"stem": p["stem"], "tf": p["tf"], "surface_forms": surface_forms(p["stem"], title, abstract)} for p in parts
        ],
        "contributions": parts,
        "bm25_score": sum(p["bm25"] + p["title_boost"] for p in parts),
        "evidence": {"class": class_id, "rule_id": rule_id, "matched_text": matched},
        "best_sentence": sentence,
        "best_sentence_missing_reason": missing_reason,
    }


def surface_forms(stem, title, abstract):
    words = (m.group(0).lower() for m in WORD_RE.finditer(f"{title} {abstract}"))
    return sorted({w for w in words if stem_term(w) == stem})


def format_explanation(result):
    lines = [f"[{result['ext_id']}] {result['title']}", f"BM25 score {result['bm25_score']:.4f}"]
    for part in result["contributions"]:
        title_note = f" (+ title boost {part['title_boost']:.4f})" if part["title_boost"] else ""
        forms = ", ".join(next(m["surface_forms"] for m in result["matched_stems"] if m["stem"] == part["stem"]))
        lines.append(f"  {part['stem']:<14} tf={part['tf']:<3} bm25={part['bm25']:.4f}{title_note}  [{forms}]")
    evidence = result["evidence"]
    lines.append(f"Evidence: {evidence['class']} (rule {evidence['rule_id']}, matched '{evidence['matched_text']}')")
    if result["best_sentence"]:
        sentence = result["best_sentence"]
        lines.append(f"Best sentence (cosine {sentence['cosine']:.3f}): {sentence['text']}")
        lines.append(f"  highlight offsets: {sentence['highlights']}")
    else:
        lines.append(f"Best sentence: none ({result['best_sentence_missing_reason']})")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        print('Usage: python -m engine.explain <ext_id> "<query>"')
        sys.exit(0)
    with open("config.json", "r") as f:
        config = json.load(f)
    index = SearchIndex()
    doc_id = index.doc_id_for(sys.argv[1])
    if doc_id is None:
        print(f"Unknown document id: {sys.argv[1]}")
        sys.exit(1)
    concepts = understand(" ".join(sys.argv[2:]), index, config)
    print(format_explanation(explain(doc_id, concepts, index, config)))


if __name__ == "__main__":
    main()
