"""
download_data.py
Downloads the TREC-COVID dataset directly from the BEIR project's public URL.
No heavy dependencies (no torch, no sentence_transformers).
Saves corpus, queries, and relevance judgments (qrels) to data/raw/beir/trec-covid/.
"""

import os
import json
import csv
import zipfile
import requests


DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
DATA_DIR = os.path.join("data", "raw", "beir")


def download_trec_covid(data_dir=DATA_DIR):
    """
    Downloads and extracts the TREC-COVID dataset.
    Returns (corpus, queries, qrels, data_path).
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "trec-covid.zip")
    extract_dir = os.path.join(data_dir, "trec-covid")

    # skip download if already extracted
    if os.path.exists(os.path.join(extract_dir, "corpus.jsonl")):
        print(f"[download] Dataset already exists at {extract_dir}")
    else:
        # download the zip file
        if not os.path.exists(zip_path):
            print(f"[download] Downloading TREC-COVID from {DATASET_URL}...")
            resp = requests.get(DATASET_URL, stream=True, timeout=300)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(f"\r  [download] {downloaded // (1024*1024)} MB / {total // (1024*1024)} MB ({pct}%)", end="", flush=True)
            print()
            print(f"[download] Saved to {zip_path}")
        else:
            print(f"[download] Zip already exists at {zip_path}")

        # extract
        print(f"[download] Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print(f"[download] Extraction complete")

    # load the data
    corpus = load_corpus(extract_dir)
    queries = load_queries(extract_dir)
    qrels = load_qrels(extract_dir)

    print(f"[download] Corpus: {len(corpus)} docs, Queries: {len(queries)}, Qrels: {len(qrels)} topics")
    return corpus, queries, qrels, extract_dir


def load_corpus(data_path):
    """
    Reads corpus.jsonl: each line is a JSON object with _id, title, text.
    Returns dict mapping doc_id -> {"title": ..., "text": ...}
    """
    corpus = {}
    corpus_file = os.path.join(data_path, "corpus.jsonl")
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
    return corpus


def load_queries(data_path):
    """
    Reads queries.jsonl: each line has _id and text.
    Returns dict mapping query_id -> query_text.
    """
    queries = {}
    queries_file = os.path.join(data_path, "queries.jsonl")
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(data_path, split="test"):
    """
    Reads qrels TSV file: query_id, corpus_id, score columns.
    Returns dict mapping query_id -> {doc_id: relevance_score}.
    """
    qrels = {}
    qrels_file = os.path.join(data_path, "qrels", f"{split}.tsv")
    with open(qrels_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            qid, did, score = row[0], row[1], int(row[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score
    return qrels


if __name__ == "__main__":
    corpus, queries, qrels, path = download_trec_covid()
    print(f"Sample doc IDs: {list(corpus.keys())[:5]}")
    print(f"Sample query: {list(queries.items())[0]}")
