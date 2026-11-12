"""
build_thesaurus.py
Trains word2vec on the stemmed titles and abstracts and saves, for each term, its nearest neighbours.

Usage: python -m pipeline.build_thesaurus

Writes index_out/thesaurus.json (stem -> [[neighbour, cosine], ...]) and index_out/word2vec.model.
Note: workers > 1 means the training is not bit-reproducible even with a fixed seed.
"""

import json
import os
import sqlite3
import sys
import time

import numpy as np
from gensim.models import Word2Vec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.tokenizer import tokenize_no_stem, get_stems_for_tokens

INDEX_DIR = "index_out"
SAMPLE_STEMS = ["covid", "vaccin", "pregnant", "mortal", "transmiss", "ace2", "remdesivir", "mask", "children", "ventil"]
NEIGHBOUR_POOL = 30
BATCH_SIZE = 2000


def load_sentences(db_path):
    """One stemmed token list per document (title and abstract), stems shared through a cache."""
    stem_cache = {}
    sentences = []
    connection = sqlite3.connect(db_path)
    for title, abstract in connection.execute("SELECT title, abstract FROM docs ORDER BY doc_id"):
        raw_tokens = tokenize_no_stem((title or "") + " " + (abstract or ""))
        for token in raw_tokens:
            if token not in stem_cache:
                stem_cache[token] = get_stems_for_tokens([token])[0]
        sentences.append([stem_cache[token] for token in raw_tokens])
    connection.close()
    return sentences


def train_model(sentences, settings):
    return Word2Vec(
        sentences,
        vector_size=settings["vector_size"],
        window=settings["window"],
        min_count=settings["min_count"],
        sg=settings["sg"],
        epochs=settings["epochs"],
        workers=settings["workers"],
        seed=settings["seed"],
    )


def build_neighbour_table(model, dictionary, settings):
    """For every eligible stem keep up to top_neighbours terms with cosine >= min_similarity."""
    min_df = settings["min_df"]
    words = [w for w in model.wv.index_to_key if dictionary.get(w, {}).get("df", 0) >= min_df and not w.isdigit()]
    positions = np.array([model.wv.key_to_index[w] for w in words])
    vectors = model.wv.vectors[positions]
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    table = {}
    for start in range(0, len(words), BATCH_SIZE):
        similarities = vectors[start:start + BATCH_SIZE] @ vectors.T
        pool = min(NEIGHBOUR_POOL, len(words))
        best = np.argpartition(-similarities, pool - 1, axis=1)[:, :pool]
        for row, candidates in enumerate(best):
            word = words[start + row]
            ranked = sorted(candidates, key=lambda c: -similarities[row, c])
            neighbours = [
                [words[c], round(float(similarities[row, c]), 4)]
                for c in ranked
                if words[c] != word and similarities[row, c] >= settings["min_similarity"]
            ][: settings["top_neighbours"]]
            if neighbours:
                table[word] = neighbours
    return table


def main():
    with open("config.json", "r") as f:
        settings = json.load(f)["thesaurus"]
    with open(os.path.join(INDEX_DIR, "dictionary.json"), "r") as f:
        dictionary = json.load(f)

    start = time.time()
    print("[thesaurus] Reading and stemming documents...")
    sentences = load_sentences(os.path.join(INDEX_DIR, "abstracts.sqlite"))
    print(f"[thesaurus] {len(sentences)} documents in {time.time() - start:.1f}s; training word2vec...")
    model = train_model(sentences, settings)
    model.save(os.path.join(INDEX_DIR, "word2vec.model"))
    table = build_neighbour_table(model, dictionary, settings)
    with open(os.path.join(INDEX_DIR, "thesaurus.json"), "w") as f:
        json.dump(table, f, separators=(",", ":"))

    print(f"[thesaurus] {len(table)} stems have neighbours. Total time {time.time() - start:.1f}s")
    for name in ("thesaurus.json", "word2vec.model"):
        print(f"  {name}: {os.path.getsize(os.path.join(INDEX_DIR, name)) / (1024 * 1024):.1f} MB")
    for stem in SAMPLE_STEMS:
        print(f"  {stem}: {table.get(stem, [])}")


if __name__ == "__main__":
    main()
