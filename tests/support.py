"""Builds a small real index (shards, title postings, abstracts) into a folder for tests."""

import json
import os

from pipeline.build_index import (
    build_inverted_index, build_kgram_index, build_title_postings, export_shards, write_abstract_store,
)

CONFIG = {"shard_max_bytes": 5000000}


def build_tiny_index(corpus, folder):
    postings, doc_meta, vocab, stats = build_inverted_index(corpus, CONFIG)
    export_shards(postings, vocab, doc_meta, build_kgram_index(vocab), stats, CONFIG, folder)
    with open(os.path.join(folder, "title_postings.json"), "w") as f:
        json.dump(build_title_postings(corpus), f)
    write_abstract_store(corpus, os.path.join(folder, "abstracts.sqlite"))
