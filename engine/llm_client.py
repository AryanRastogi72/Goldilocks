"""Send query proposal prompts to Groq and cache successful text responses."""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

import requests


ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT, "llm_cache")
_session_api_key = ContextVar("goldilocks_api_key", default=None)
RATE_LIMIT_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def load_config():
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)


@contextmanager
def using_api_key(api_key):
    token = _session_api_key.set(api_key)
    try:
        yield
    finally:
        _session_api_key.reset(token)


def _cache_path(prompt):
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.json")


def _get_cached(prompt):
    path = _cache_path(prompt)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)["response"]


def _set_cache(prompt, response):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(prompt), "w", encoding="utf-8") as file:
        json.dump({"prompt": prompt, "response": response, "timestamp": time.time()}, file)


def call_model(prompt, use_cache=True):
    if use_cache:
        cached = _get_cached(prompt)
        if cached is not None:
            return cached

    api_key = _session_api_key.get() or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    model = load_config().get("groq_model", "openai/gpt-oss-20b")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_completion_tokens": 1024,
        "reasoning_effort": "low",
    }

    try:
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                time.sleep(min(2 ** attempt, 10))
                continue

            if response.status_code not in RETRYABLE_STATUS_CODES or attempt == RATE_LIMIT_RETRIES:
                break
            try:
                delay = float(response.headers.get("Retry-After", 2 ** attempt))
            except (TypeError, ValueError):
                delay = 2 ** attempt
            time.sleep(min(max(delay, 0), 30))
        response.raise_for_status()
        choices = response.json().get("choices", [])
        text = choices[0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Groq response contained no text.")
        if use_cache:
            _set_cache(prompt, text)
        return text.strip()
    except requests.exceptions.RequestException as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status is not None:
            raise RuntimeError(f"Groq request failed with HTTP {status}.") from None
        raise RuntimeError("Groq request failed. Check the connection and model settings.") from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError("Groq response format was unexpected.") from None


def propose_boolean_query(user_query, context=""):
    prompt = f"""You are helping build a Boolean search query for a biomedical document collection about COVID 19.

The user wants to find documents about: "{user_query}"
{f"Additional context: {context}" if context else ""}

Propose a Boolean query using AND, OR, NOT operators and parentheses.
Use specific medical and scientific terms. Keep it focused with 3 to 8 terms.
Return only the Boolean query string, nothing else.

Example format: covid AND (treatment OR therapy) AND clinical"""
    return call_model(prompt)


def suggest_synonyms(term, context=""):
    prompt = f"""Suggest 3 to 5 synonyms or closely related terms for "{term}" in biomedical research about COVID 19.
{f"Search context: {context}" if context else ""}

Return only a comma separated list of terms, nothing else.
Example: treatment, therapy, intervention, medication"""
    response = call_model(prompt)
    return [item.strip().lower() for item in response.split(",") if item.strip()]


def refine_query(current_query, feedback, user_query=""):
    prompt = f"""You are refining a Boolean search query for a biomedical document collection about COVID 19.

Original user question: "{user_query}"
Current Boolean query: {current_query}
Feedback: {feedback}

Propose an improved Boolean query. Use AND, OR, NOT operators.
Return only the Boolean query string, nothing else."""
    return call_model(prompt)
