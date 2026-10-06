"""
gemini_client.py
Client for the Google Gemini API with disk-based response caching.
The LLM only proposes search terms and synonyms. It never makes
retrieval decisions; those come from the controller using index statistics.
"""

import os
import sys
import json
import hashlib
import time
import requests
from contextvars import ContextVar
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


# directory for caching Gemini responses to avoid redundant API calls
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "gemini_cache")
_session_api_key = ContextVar("goldilocks_api_key", default=None)


@contextmanager
def using_api_key(api_key):
    token = _session_api_key.set(api_key)
    try:
        yield
    finally:
        _session_api_key.reset(token)


def _cache_key(prompt):
    """Creates a deterministic cache key from the prompt text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _get_cached(prompt):
    """Returns cached response if available, else None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(prompt)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)["response"]
    return None


def _set_cache(prompt, response):
    """Saves a response to the disk cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(prompt)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "response": response, "timestamp": time.time()}, f)


def call_gemini(prompt, use_cache=True):
    """
    Calls the Gemini API with the given prompt.
    Uses disk caching to avoid redundant calls (important for evaluation
    reruns and for staying within free tier rate limits).

    Returns the text response from the model.
    Raises RuntimeError if the API call fails.
    """
    # check cache first
    if use_cache:
        cached = _get_cached(prompt)
        if cached is not None:
            return cached

    config = load_config()
    model = config.get("gemini_model", "gemini-3.8-flash")

    # API key from environment variable (never hardcoded)
    api_key = _session_api_key.get() or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key from https://aistudio.google.com/"
        )

    # Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,  # low temperature for more deterministic output
            "maxOutputTokens": 512,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # extract text from response
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini response contained no candidates.")

        try:
            text = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("Gemini response format was unexpected.") from None

        # cache the successful response
        if use_cache:
            _set_cache(prompt, text)

        return text

    except requests.exceptions.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status is not None:
            raise RuntimeError(f"Gemini API call failed with HTTP {status}.") from None
        raise RuntimeError("Gemini API request failed. Check the connection and API settings.") from None


def propose_boolean_query(user_query, context=""):
    """
    Asks Gemini to propose a Boolean query for a natural language question.
    The prompt instructs the model to use AND, OR, NOT operators and suggest
    relevant medical/scientific terms.
    """
    prompt = f"""You are helping build a Boolean search query for a biomedical document collection about COVID-19.

The user wants to find documents about: "{user_query}"
{f"Additional context: {context}" if context else ""}

Propose a Boolean query using AND, OR, NOT operators and parentheses.
Use specific medical and scientific terms. Keep it focused (3 to 8 terms).
Return ONLY the Boolean query string, nothing else.

Example format: covid AND (treatment OR therapy) AND clinical"""

    return call_gemini(prompt)


def suggest_synonyms(term, context=""):
    """
    Asks Gemini to suggest synonyms or related terms for expansion.
    Used by the controller when the result set is too small.
    """
    prompt = f"""Suggest 3 to 5 synonyms or closely related terms for "{term}" in the context of biomedical research about COVID-19.
{f"Search context: {context}" if context else ""}

Return ONLY a comma-separated list of terms, nothing else.
Example: treatment, therapy, intervention, medication"""

    response = call_gemini(prompt)
    # parse comma-separated list
    synonyms = [s.strip().lower() for s in response.split(",") if s.strip()]
    return synonyms


def refine_query(current_query, feedback, user_query=""):
    """
    Asks Gemini to refine a Boolean query based on feedback from the controller.
    The feedback tells the LLM what went wrong (too many hits, too few, etc.)
    and the model proposes adjustments.
    """
    prompt = f"""You are refining a Boolean search query for a biomedical document collection about COVID-19.

Original user question: "{user_query}"
Current Boolean query: {current_query}
Feedback: {feedback}

Propose an improved Boolean query. Use AND, OR, NOT operators.
Return ONLY the Boolean query string, nothing else."""

    return call_gemini(prompt)


if __name__ == "__main__":
    # test with a sample query (requires GEMINI_API_KEY)
    try:
        result = propose_boolean_query("What drugs are effective against COVID-19?")
        print(f"Proposed query: {result}")
    except RuntimeError as e:
        print(f"Error: {e}")
