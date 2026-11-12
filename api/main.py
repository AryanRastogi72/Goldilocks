"""
main.py
The EvidenceLens API. Loads the index once at startup and serves search, follow-ups, clarification,
documents with explanations, the saved evaluation data and a health check.

Run: uvicorn api.main:app   (port 8000)
"""

import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Path as PathParam, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.eval_data import build_eval_payload
from api.schemas import SESSION_ID_PATTERN, ClarifyRequest, SearchRequest
from engine.controller import load_config
from engine.conversation import KEEP_FILTER, SESSIONS, apply_clarification, handle_turn
from engine.explain import explain
from engine.search import SearchIndex

API_VERSION = "0.6.0"
MAX_SESSIONS = 500
IDLE_SESSION_SECONDS = 60 * 60
DEFAULT_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
DEV_ORIGIN = "http://localhost:5173"
WARM_UP_QUERY = "covid"
LOG = logging.getLogger("uvicorn.error.evidencelens")


class ApiError(Exception):
    def __init__(self, status_code, code, message):
        self.status_code, self.code, self.message = status_code, code, message


def error_response(status_code, code, message):
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def normalise_title(title):
    return re.sub(r"[^a-z0-9]", "", title.lower())


def group_duplicates(items):
    """
    Display-only grouping: items whose normalised titles are identical collapse into the first one, which lists
    the others in `duplicates` (ext_ids). Items with an empty normalised title are never grouped.
    Evaluation code must not use this.
    """
    groups, by_title = [], {}
    for item in items:
        key = normalise_title(item["title"]) or item["ext_id"]
        if key in by_title:
            by_title[key]["duplicates"].append(item["ext_id"])
        else:
            by_title[key] = dict(item, duplicates=[])
            groups.append(by_title[key])
    return groups


def concept_view(concept, status):
    keys = ("label", "stems", "idf", "required", "excluded", "source", "facet")
    return dict({key: concept[key] for key in keys if key in concept}, status=status)


def matched_terms(explanation):
    forms = {m["stem"]: m["surface_forms"] for m in explanation["matched_stems"]}
    return [
        {"stem": p["stem"], "surface_forms": forms[p["stem"]], "tf": p["tf"], "bm25": p["bm25"], "title_boost": p["title_boost"]}
        for p in explanation["contributions"]
    ]


def build_results(result, index, config):
    """Adds the explanation to every result and groups duplicate titles. Returns (results, explain_ms)."""
    started = time.perf_counter()
    items = []
    for item in result["results"]:
        explanation = explain(item["doc_id"], [], index, config, extra_stems=result["ranking_stems"])
        sentence = explanation["best_sentence"]
        items.append({
            "doc_id": item["doc_id"], "ext_id": item["ext_id"], "title": item["title"], "score": item["score"],
            "evidence_class": explanation["evidence"]["class"], "evidence_rule": explanation["evidence"]["rule_id"],
            "matched_terms": matched_terms(explanation),
            "best_sentence": sentence["text"] if sentence else None,
            "highlights": sentence["highlights"] if sentence else [],
        })
    grouped = group_duplicates(items)
    for rank, item in enumerate(grouped, start=1):
        item["rank"] = rank
    return grouped, (time.perf_counter() - started) * 1000


def build_response(session_id, result, config, index, started, controller_ms):
    results, explain_ms = build_results(result, index, config)
    low, high = config["target_band_low"], config["target_band_high"]
    return {
        "session_id": session_id,
        "rewritten_query": result["rewritten_query"],
        "resolution_rules": result["rules_fired"],
        "boolean_query": result["boolean_query"],
        "concepts": [concept_view(c, result["concept_statuses"][c["label"]]) for c in result["concepts"]],
        "trace": result["trace"],
        "clarifications": result["clarifications"],
        "results": results,
        "hits": result["hits"],
        "in_band": low <= result["hits"] <= high if result["ranker"] == "agent" else None,
        "band": [low, high],
        "evidence_filter": result["evidence_filter"],
        "ranker": result["ranker"],
        "timings": {
            "total_ms": (time.perf_counter() - started) * 1000,
            "controller_ms": controller_ms,
            "explain_ms": explain_ms,
        },
    }


def validation_message(error):
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"] if part not in ("body", "query", "path"))
        problems.append(f"{location}: {item['msg']}" if location else item["msg"])
    return "; ".join(problems)


def register_error_handlers(app):
    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError):
        return error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return error_response(422, "invalid_request", validation_message(error))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        code = "not_found" if error.status_code == 404 else "http_error"
        return error_response(error.status_code, code, str(error.detail))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        LOG.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", "Something went wrong on the server. The details are in the server log.")


def register_static_files(app, dist_dir):
    if dist_dir is None or not Path(dist_dir).is_dir():
        @app.get("/")
        def root():
            return {"message": "EvidenceLens API. The web app is not built yet; see /api/health.", "health": "/api/health"}
        return
    dist = Path(dist_dir).resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def single_page_app(path: str):
        if path.startswith("api/"):
            raise StarletteHTTPException(404, "Unknown API route")
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and dist in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


def warm_up(state):
    """One throwaway search at startup so the first real request does not pay for loading shards."""
    started = time.perf_counter()
    result = handle_turn("_warm_up", WARM_UP_QUERY, state.index, state.config, allow_clarify=False)
    build_results(result, state.index, state.config)
    SESSIONS.pop("_warm_up")
    LOG.info("Warm-up search finished in %.0f ms", (time.perf_counter() - started) * 1000)


def create_app(dist_dir=DEFAULT_DIST, load_engine=True):
    """Builds the app. load_engine=False skips loading the index (for tests of routes that do not need it)."""

    @asynccontextmanager
    async def lifespan(app):
        if load_engine:
            app.state.config = load_config()
            app.state.index = SearchIndex()
            app.state.eval_payload = build_eval_payload()
            SESSIONS.configure(MAX_SESSIONS, IDLE_SESSION_SECONDS)
            warm_up(app.state)
        yield
        if load_engine:
            app.state.index.close()

    app = FastAPI(title="EvidenceLens API", version=API_VERSION, lifespan=lifespan)
    app.state.engine_lock = threading.Lock()
    app.add_middleware(CORSMiddleware, allow_origins=[DEV_ORIGIN], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    register_error_handlers(app)

    @app.get("/api/health")
    def health(request: Request):
        index = getattr(request.app.state, "index", None)
        return {
            "status": "ok" if index else "starting", "docs": index.num_docs if index else 0,
            "terms": len(index.dictionary) if index else 0, "index_loaded": index is not None, "version": API_VERSION,
        }

    @app.post("/api/search")
    def search(body: SearchRequest, request: Request):
        state = request.app.state
        session_id = body.session_id or uuid.uuid4().hex[:16]
        options = {"evidence_filter": body.evidence_filter} if "evidence_filter" in body.model_fields_set else {}
        with state.engine_lock:
            new_session = body.session_id is not None and body.session_id not in SESSIONS
            started = time.perf_counter()
            result = handle_turn(session_id, body.text, state.index, state.config, allow_clarify=body.ranker == "agent", ranker=body.ranker, **options)
            controller_ms = (time.perf_counter() - started) * 1000
            response = build_response(session_id, result, state.config, state.index, started, controller_ms)
            response["new_session"] = new_session
            return response

    @app.post("/api/clarify")
    def clarify(body: ClarifyRequest, request: Request):
        state = request.app.state
        with state.engine_lock:
            if body.session_id not in SESSIONS:
                raise ApiError(404, "unknown_session", "This session does not exist or has expired. Start a new search.")
            choices = SESSIONS[body.session_id]["clarifications"]
            if body.choice_id not in [c["id"] for c in choices]:
                raise ApiError(422, "unknown_choice", f"choice_id {body.choice_id} is not among the clarification choices of the last turn.")
            started = time.perf_counter()
            result = apply_clarification(body.session_id, body.choice_id, state.index, state.config)
            controller_ms = (time.perf_counter() - started) * 1000
            return build_response(body.session_id, result, state.config, state.index, started, controller_ms)

    @app.get("/api/doc/{ext_id}")
    def document(request: Request, ext_id: str = PathParam(pattern=SESSION_ID_PATTERN),
                 session_id: str | None = Query(default=None, pattern=SESSION_ID_PATTERN)):
        state = request.app.state
        with state.engine_lock:
            doc_id = state.index.doc_id_for(ext_id)
            if doc_id is None:
                raise ApiError(404, "unknown_document", f"No document with id {ext_id}.")
            session = SESSIONS[session_id] if session_id and session_id in SESSIONS else None
            stems = session.get("ranking_stems", []) if session else []
            title, abstract = state.index.get_title_and_abstract(doc_id)
            explanation = explain(doc_id, [], state.index, state.config, extra_stems=stems)
        return {"ext_id": ext_id, "doc_id": doc_id, "title": title, "abstract": abstract,
                "session_used": session is not None, "explanation": explanation}

    @app.get("/api/eval")
    def evaluation(request: Request):
        return request.app.state.eval_payload

    register_static_files(app, dist_dir)
    return app


app = create_app()
