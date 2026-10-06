"""
search.py
Core search operations: loading the index, Boolean query execution with
df-ordered intersection, and wildcard expansion via the k-gram index.
"""

import os
import sys
import json
import math
import re
from collections import defaultdict, OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.tokenizer import stem_term


class SearchIndex:
    """
    Loads and manages the inverted index built by Phase 1.
    Supports:
      - Looking up postings for a term
      - Boolean query execution (AND, OR, NOT)
      - df-ordered intersection (process smallest postings list first)
      - Wildcard expansion using the k-gram index
    """

    def __init__(self, index_dir="index_out", max_cached_shards=None):
        self.index_dir = index_dir
        self.max_cached_shards = max_cached_shards

        # load dictionary: term -> {df, shard}
        with open(os.path.join(index_dir, "dictionary.json"), "r") as f:
            self.dictionary = json.load(f)

        # load doc metadata: int_id -> {ext_id, title, norm}
        with open(os.path.join(index_dir, "doc_meta.json"), "r") as f:
            self.doc_meta = json.load(f)

        # load k-gram index: gram -> [term1, term2, ...]
        with open(os.path.join(index_dir, "kgram_index.json"), "r") as f:
            self.kgram_index = json.load(f)

        # load corpus stats
        with open(os.path.join(index_dir, "corpus_stats.json"), "r") as f:
            self.corpus_stats = json.load(f)

        self.num_docs = self.corpus_stats["num_docs"]

        # cache for loaded shards (shard_id -> {term: [[doc_id, tf], ...]})
        self._shard_cache = OrderedDict()

    def get_df(self, term):
        """Returns document frequency for a term, or 0 if not in vocabulary."""
        entry = self.dictionary.get(term)
        if entry is None:
            return 0
        return entry["df"]

    def get_postings(self, term):
        """
        Returns the postings list for a term as document and term count pairs.
        Loads the shard from disk if not already cached.
        """
        entry = self.dictionary.get(term)
        if entry is None:
            return []

        shard_id = entry["shard"]

        # load shard if not cached
        if shard_id not in self._shard_cache:
            shard_path = os.path.join(self.index_dir, f"postings_{shard_id:03d}.json")
            with open(shard_path, "r") as f:
                self._shard_cache[shard_id] = json.load(f)
            if self.max_cached_shards is not None:
                while len(self._shard_cache) > max(1, self.max_cached_shards):
                    self._shard_cache.popitem(last=False)
        else:
            self._shard_cache.move_to_end(shard_id)

        shard_data = self._shard_cache[shard_id]
        raw = shard_data.get(term, [])
        return raw

    def get_idf(self, term):
        """Computes idf = log10(N / df) for a term."""
        df = self.get_df(term)
        if df == 0:
            return 0
        return math.log10(self.num_docs / df)

    def get_doc_norm(self, doc_id):
        """Returns the precomputed L2 norm for a document."""
        meta = self.doc_meta.get(str(doc_id))
        if meta is None:
            return 1.0  # fallback to avoid division by zero
        return meta.get("norm", 1.0)

    def get_doc_title(self, doc_id):
        """Returns the title of a document given its internal integer ID."""
        meta = self.doc_meta.get(str(doc_id))
        if meta is None:
            return "(unknown)"
        return meta.get("title", "(no title)")

    def get_ext_id(self, doc_id):
        """Returns the external (TREC) document ID given the internal integer ID."""
        meta = self.doc_meta.get(str(doc_id))
        if meta is None:
            return str(doc_id)
        return meta.get("ext_id", str(doc_id))

    # ---- Boolean execution ----

    def execute_boolean(self, ast):
        """
        Executes a Boolean AST against the index.
        Returns a set of matching document IDs (integers).

        For AND, we use df-ordered intersection: sort children by their
        postings list size (smallest first) and intersect incrementally.
        This minimizes comparisons, a key optimization from the lectures.
        """
        op = ast[0]

        if op == "TERM":
            term = ast[1]
            postings = self.get_postings(term)
            return set(doc_id for doc_id, tf in postings)

        elif op == "WILDCARD":
            prefix = ast[1]
            expanded_terms = self.expand_wildcard(prefix)
            # OR all expanded terms together
            result = set()
            for term in expanded_terms:
                postings = self.get_postings(term)
                result |= set(doc_id for doc_id, tf in postings)
            return result

        elif op == "AND":
            children = ast[1]
            # get result sets for each child, ordered by size (smallest first)
            child_results = []
            for child in children:
                child_results.append(self.execute_boolean(child))

            # sort by set size for efficient intersection
            child_results.sort(key=len)

            # incremental intersection: start with smallest set
            result = child_results[0]
            for other in child_results[1:]:
                result = result & other
                # early termination: if result is empty, no need to continue
                if not result:
                    break
            return result

        elif op == "OR":
            children = ast[1]
            result = set()
            for child in children:
                result |= self.execute_boolean(child)
            return result

        elif op == "NOT":
            child = ast[1]
            child_result = self.execute_boolean(child)
            # NOT returns all docs except those matching the child
            all_docs = set(range(self.num_docs))
            return all_docs - child_result

        else:
            raise ValueError(f"Unknown AST node type: {op}")

    # ---- Wildcard expansion ----

    def expand_wildcard(self, prefix):
        """
        Expands a wildcard prefix (e.g., "treat") into matching terms
        using the k-gram index.

        Strategy: generate k-grams from the prefix (with $ start marker),
        intersect the term sets for each k-gram, then filter by prefix match.
        """
        k = 3  # trigram
        padded = f"${prefix}"

        # generate k-grams from the prefix
        grams = []
        for i in range(len(padded) - k + 1):
            grams.append(padded[i : i + k])

        if not grams:
            # prefix too short for k-grams, fall back to linear scan
            return [t for t in self.dictionary if t.startswith(prefix)]

        # intersect term sets for all k-grams
        candidates = None
        for gram in grams:
            gram_terms = set(self.kgram_index.get(gram, []))
            if candidates is None:
                candidates = gram_terms
            else:
                candidates = candidates & gram_terms

        if candidates is None:
            return []

        # filter by actual prefix match (k-grams can produce false positives)
        matches = [t for t in candidates if t.startswith(prefix)]
        return sorted(matches)

    # ---- Hit estimation ----

    def estimate_hits(self, ast):
        """
        Estimates the number of hits for a Boolean query without executing it.
        AND: at most min(df of children)  (upper bound)
        OR:  at most sum(df of children)  (upper bound, overcounts overlaps)
        NOT: num_docs minus child hits (approximate)
        TERM: exactly df

        These bounds help the controller decide whether to tighten or relax.
        """
        op = ast[0]

        if op == "TERM":
            return self.get_df(ast[1])

        elif op == "WILDCARD":
            prefix = ast[1]
            expanded = self.expand_wildcard(prefix)
            return sum(self.get_df(t) for t in expanded)

        elif op == "AND":
            children = ast[1]
            child_estimates = [self.estimate_hits(c) for c in children]
            # AND cannot return more than the smallest child
            return min(child_estimates) if child_estimates else 0

        elif op == "OR":
            children = ast[1]
            child_estimates = [self.estimate_hits(c) for c in children]
            # OR can return at most the sum (upper bound, ignores overlap)
            return sum(child_estimates)

        elif op == "NOT":
            child_est = self.estimate_hits(ast[1])
            return max(0, self.num_docs - child_est)

        return 0

    def get_df_table(self, ast):
        """
        Builds a table of term -> df for all terms in the AST.
        Used for logging in the controller loop.
        """
        table = {}
        self._collect_df(ast, table)
        return table

    def _collect_df(self, ast, table):
        """Recursively collects df for all terms in the AST."""
        op = ast[0]
        if op == "TERM":
            table[ast[1]] = self.get_df(ast[1])
        elif op == "WILDCARD":
            prefix = ast[1]
            expanded = self.expand_wildcard(prefix)
            for t in expanded:
                table[t] = self.get_df(t)
            table[f"{prefix}*"] = sum(self.get_df(t) for t in expanded)
        elif op in ("AND", "OR"):
            for child in ast[1]:
                self._collect_df(child, table)
        elif op == "NOT":
            self._collect_df(ast[1], table)


if __name__ == "__main__":
    # quick test if index exists
    if os.path.exists("index_out/dictionary.json"):
        idx = SearchIndex()
        print(f"Loaded index: {idx.num_docs} docs, {len(idx.dictionary)} terms")
        # test a few terms
        for term in ["covid", "treatment", "vaccin"]:
            stemmed = stem_term(term)
            df = idx.get_df(stemmed)
            print(f"  '{term}' -> stem '{stemmed}', df={df}")
    else:
        print("Index not built yet. Run: python -m pipeline.build_index")
