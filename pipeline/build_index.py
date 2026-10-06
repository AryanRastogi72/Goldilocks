"""
build_index.py
Offline pipeline that builds the inverted index, k-gram index, and exports
compact JSON shards for the browser engine.

Usage: python -m pipeline.build_index

What it produces (in index_out/):
  - postings_NNN.json: sharded postings lists (term -> list of [doc_id_int, tf])
  - dictionary.json: term -> {df, shard_id, offset}  (the dictionary for lookups)
  - doc_meta.json: int_id -> {ext_id, title, norm}  (doc metadata and L2 norms)
  - kgram_index.json: k-gram -> list of terms containing that k-gram
  - corpus_stats.json: {num_docs, avg_dl, total_terms}
"""

import os
import sys
import json
import math
import time
from collections import defaultdict, Counter

# add project root to path so we can import sibling packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.tokenizer import tokenize, stem_term


def load_config():
    """Reads config.json from project root."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def build_inverted_index(corpus, config):
    """
    Builds the inverted index from the corpus.

    For each document, tokenize the title + text, count term frequencies,
    and build postings lists. Also computes document norms for tf-idf cosine.

    corpus: dict mapping doc_id (string) -> {"title": ..., "text": ...}
    Returns: (postings, doc_meta, vocab, corpus_stats)
      - postings: dict term -> list of (int_doc_id, tf)
      - doc_meta: dict int_doc_id -> {"ext_id": str, "title": str, "norm": float}
      - vocab: dict term -> {"df": int} (document frequency)
      - corpus_stats: {"num_docs": int, "avg_dl": float, "total_terms": int}
    """
    print("[index] Building inverted index...")
    t0 = time.time()

    # map external doc IDs (strings) to sequential integers for compact storage
    ext_to_int = {}
    doc_meta = {}
    postings = defaultdict(list)  # term -> [(int_doc_id, tf), ...]
    vocab = {}  # term -> {"df": int}

    total_tokens = 0
    num_docs = len(corpus)

    for i, (doc_id, doc) in enumerate(corpus.items()):
        # assign sequential integer ID
        int_id = i
        ext_to_int[doc_id] = int_id

        # combine title and text for indexing
        text = (doc.get("title", "") or "") + " " + (doc.get("text", "") or "")
        tokens = tokenize(text)
        total_tokens += len(tokens)

        # count term frequency in this document
        tf_counter = Counter(tokens)

        # store doc metadata (title truncated to save space)
        title = doc.get("title", "")
        if len(title) > 200:
            title = title[:200] + "..."
        doc_meta[int_id] = {
            "ext_id": doc_id,
            "title": title,
            # norm will be computed in a second pass after idf is known
        }

        # add to postings lists
        for term, tf in tf_counter.items():
            postings[term].append((int_id, tf))

        if (i + 1) % 10000 == 0:
            print(f"  [index] Processed {i + 1}/{num_docs} documents...")

    # compute df (document frequency) for each term
    for term, posting_list in postings.items():
        vocab[term] = {"df": len(posting_list)}

    # compute idf and document norms for tf-idf cosine similarity
    # norm = sqrt(sum of (tf * idf)^2) for all terms in the document
    print("[index] Computing document norms...")
    doc_tfidf_sq = defaultdict(float)  # int_doc_id -> sum of (tf*idf)^2

    for term, posting_list in postings.items():
        df = vocab[term]["df"]
        # idf = log10(N / df), standard formula from lecture
        idf = math.log10(num_docs / df) if df > 0 else 0
        for int_id, tf in posting_list:
            # tf weight = 1 + log10(tf) if tf > 0, else 0
            tf_weight = (1 + math.log10(tf)) if tf > 0 else 0
            doc_tfidf_sq[int_id] += (tf_weight * idf) ** 2

    # store the L2 norm in doc_meta
    for int_id in doc_meta:
        sq = doc_tfidf_sq.get(int_id, 0)
        doc_meta[int_id]["norm"] = round(math.sqrt(sq), 6)

    avg_dl = total_tokens / num_docs if num_docs > 0 else 0
    corpus_stats = {
        "num_docs": num_docs,
        "avg_dl": round(avg_dl, 2),
        "total_terms": total_tokens,
        "vocab_size": len(vocab),
    }

    elapsed = time.time() - t0
    print(f"[index] Built index in {elapsed:.1f}s: {num_docs} docs, {len(vocab)} terms")
    return postings, doc_meta, vocab, corpus_stats


def build_kgram_index(vocab, k=3):
    """
    Builds a k-gram index from the vocabulary.
    For each term, generate all k-grams (with $ as start/end markers)
    and map each k-gram to the set of terms containing it.
    Used for wildcard query expansion (e.g., "treat*" -> find all terms
    whose k-grams match).

    Returns: dict k-gram_string -> list of terms
    """
    print(f"[kgram] Building {k}-gram index...")
    t0 = time.time()
    kgram_index = defaultdict(set)

    for term in vocab:
        # add start/end markers: "$hello$" -> trigrams "$he", "hel", "ell", "llo", "lo$"
        padded = f"${term}$"
        for i in range(len(padded) - k + 1):
            gram = padded[i : i + k]
            kgram_index[gram].add(term)

    # convert sets to sorted lists for JSON serialization
    kgram_dict = {gram: sorted(terms) for gram, terms in kgram_index.items()}

    elapsed = time.time() - t0
    print(f"[kgram] Built {k}-gram index in {elapsed:.1f}s: {len(kgram_dict)} grams")
    return kgram_dict


def export_shards(postings, vocab, doc_meta, kgram_index, corpus_stats, config, out_dir="index_out"):
    """
    Exports the index as JSON shards, each under shard_max_bytes.

    Shard format:
    - Each shard is a JSON object: {term: [[doc_id, tf], [doc_id, tf], ...], ...}
    - Terms are sorted alphabetically and packed into shards until size limit.
    - dictionary.json maps each term to its shard number.

    This format lets the browser engine load only the shards it needs
    (lazy loading), keeping memory usage low.
    """
    print("[export] Exporting index shards...")
    os.makedirs(out_dir, exist_ok=True)
    max_bytes = config.get("shard_max_bytes", 20_000_000)

    # sort terms alphabetically for predictable shard assignment
    sorted_terms = sorted(postings.keys())

    # dictionary: term -> {df, shard_id}
    dictionary = {}
    current_shard = {}
    current_size = 2  # account for "{}" wrapper
    shard_id = 0

    def flush_shard():
        """Write current shard to disk and reset."""
        nonlocal current_shard, current_size, shard_id
        shard_path = os.path.join(out_dir, f"postings_{shard_id:03d}.json")
        with open(shard_path, "w") as f:
            json.dump(current_shard, f, separators=(",", ":"))
        size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        print(f"  [export] Shard {shard_id}: {len(current_shard)} terms, {size_mb:.1f} MB")
        shard_id += 1
        current_shard = {}
        current_size = 2

    for term in sorted_terms:
        # each posting entry is [doc_id, tf] to save space vs objects
        posting_list = postings[term]
        # estimate JSON size: term key + array of [int, int] pairs
        entry_str = json.dumps({term: posting_list}, separators=(",", ":"))
        entry_size = len(entry_str.encode("utf-8"))

        # if adding this term would exceed shard limit, flush
        if current_size + entry_size > max_bytes and current_shard:
            flush_shard()

        current_shard[term] = posting_list
        current_size += entry_size

        # record which shard this term lives in
        dictionary[term] = {
            "df": vocab[term]["df"],
            "shard": shard_id,
        }

    # flush remaining terms
    if current_shard:
        flush_shard()

    # write dictionary (term -> df + shard mapping)
    dict_path = os.path.join(out_dir, "dictionary.json")
    with open(dict_path, "w") as f:
        json.dump(dictionary, f, separators=(",", ":"))
    dict_size = os.path.getsize(dict_path) / (1024 * 1024)
    print(f"  [export] Dictionary: {len(dictionary)} terms, {dict_size:.1f} MB")

    # write doc metadata (int_id -> ext_id, title, norm)
    meta_path = os.path.join(out_dir, "doc_meta.json")
    with open(meta_path, "w") as f:
        json.dump(doc_meta, f, separators=(",", ":"))
    meta_size = os.path.getsize(meta_path) / (1024 * 1024)
    print(f"  [export] Doc meta: {len(doc_meta)} docs, {meta_size:.1f} MB")

    # write k-gram index
    kgram_path = os.path.join(out_dir, "kgram_index.json")
    with open(kgram_path, "w") as f:
        json.dump(kgram_index, f, separators=(",", ":"))
    kgram_size = os.path.getsize(kgram_path) / (1024 * 1024)
    print(f"  [export] K-gram index: {len(kgram_index)} grams, {kgram_size:.1f} MB")

    # write corpus stats
    stats_path = os.path.join(out_dir, "corpus_stats.json")
    with open(stats_path, "w") as f:
        json.dump(corpus_stats, f, indent=2)

    print(f"[export] Done. {shard_id} shards written to {out_dir}/")
    return shard_id


def main():
    """
    Full offline pipeline: download data -> tokenize -> build index -> export shards.
    """
    config = load_config()

    # Step 1: Download/load TREC-COVID data
    print("=" * 60)
    print("PHASE 1: Offline Index Pipeline")
    print("=" * 60)

    from pipeline.download_data import download_trec_covid
    corpus, queries, qrels, data_path = download_trec_covid()

    # Save queries and qrels for later use by evaluation
    os.makedirs("data/cache", exist_ok=True)
    with open("data/cache/queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
    with open("data/cache/qrels.json", "w", encoding="utf-8") as f:
        # qrels values may be ints, which is fine for JSON
        json.dump(qrels, f, indent=2)
    print(f"[cache] Saved {len(queries)} queries and {len(qrels)} qrels to data/cache/")

    # Step 2: Build inverted index
    postings, doc_meta, vocab, corpus_stats = build_inverted_index(corpus, config)

    # Step 3: Build k-gram index
    k = config.get("kgram_k", 3)
    kgram_index = build_kgram_index(vocab, k=k)

    # Step 4: Export shards
    num_shards = export_shards(postings, vocab, doc_meta, kgram_index, corpus_stats, config)

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Index in index_out/ ({num_shards} shards)")
    print(f"Corpus: {corpus_stats['num_docs']} docs, {corpus_stats['vocab_size']} unique terms")
    print(f"Average document length: {corpus_stats['avg_dl']} tokens")
    print("=" * 60)


if __name__ == "__main__":
    main()
