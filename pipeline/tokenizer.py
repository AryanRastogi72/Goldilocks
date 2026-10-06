"""
tokenizer.py
Text processing pipeline: tokenize, normalize (lowercase, remove punctuation),
and stem using Porter Stemmer.
"""

import re
import nltk
from nltk.stem import PorterStemmer

# download the punkt tokenizer data if not present
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# single shared stemmer instance, avoids re-creating it for every call
_stemmer = PorterStemmer()

# regex to match tokens: sequences of alphanumeric characters
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """
    Converts raw text into a list of stemmed tokens.
    Steps: lowercase -> extract alphanumeric runs -> Porter stem each.
    Returns a list of strings (stems), preserving order and duplicates
    (needed for term frequency counting).
    """
    # lowercase the entire text first
    lower = text.lower()
    # extract all runs of alphanumeric characters (drops punctuation, whitespace)
    raw_tokens = _TOKEN_RE.findall(lower)
    # stem each token with Porter stemmer
    stemmed = [_stemmer.stem(t) for t in raw_tokens]
    return stemmed


def tokenize_no_stem(text):
    """
    Same as tokenize but skips stemming. Used to get the raw normalized
    tokens for k-gram index building (we build k-grams on stems, but
    this is useful for debugging).
    """
    lower = text.lower()
    return _TOKEN_RE.findall(lower)


def stem_term(term):
    """
    Stems a single term. Used when processing query terms at search time.
    """
    return _stemmer.stem(term.lower())


def get_stems_for_tokens(tokens):
    """
    Takes a list of raw tokens (already lowercased) and returns their stems.
    Useful when you need both the raw token and its stem.
    """
    return [_stemmer.stem(t) for t in tokens]


if __name__ == "__main__":
    # quick test
    sample = "COVID-19 vaccines: mRNA-based treatments for SARS-CoV-2"
    print(f"Input: {sample}")
    print(f"Tokens: {tokenize(sample)}")
    print(f"No stem: {tokenize_no_stem(sample)}")
