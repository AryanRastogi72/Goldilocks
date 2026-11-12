"""
conversation.py
Multi-turn sessions. Each turn is rewritten into a full set of concepts by five named rules,
the controller runs on those concepts, and the rewritten query plus the rules that fired are returned.

Rules, in the order they are checked:
  EVIDENCE_FILTER, EXCLUSION, FOLLOWUP_DETECT, FACET_SWAP, TOPIC_SWITCH.
"""

import re
import threading
import time
from collections import OrderedDict

from engine.controller import ranking_stems, run_controller
from engine.evidence import class_of, docs_of_class, g_value, validate_class
from engine.query_understanding import make_concept, mark_elimination, understand
from engine.ranker import apply_evidence_weight, rank_bm25

KEEP_FILTER = object()
BM25_DEPTH = 1000


class SessionStore:
    """
    In-memory sessions. At most max_sessions are kept; a session idle for idle_seconds expires; when the store
    is full the least recently used session is evicted first. All access goes through one lock.
    """

    def __init__(self, max_sessions=500, idle_seconds=3600, clock=time.monotonic):
        self.max_sessions, self.idle_seconds, self.clock = max_sessions, idle_seconds, clock
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def configure(self, max_sessions, idle_seconds):
        with self._lock:
            self.max_sessions, self.idle_seconds = max_sessions, idle_seconds

    def _expire(self):
        now = self.clock()
        for session_id in [s for s, (_, last_used) in self._items.items() if now - last_used > self.idle_seconds]:
            del self._items[session_id]

    def setdefault(self, session_id, default):
        with self._lock:
            self._expire()
            if session_id not in self._items:
                self._items[session_id] = [default, self.clock()]
                while len(self._items) > self.max_sessions:
                    self._items.popitem(last=False)
            return self._touch(session_id)

    def _touch(self, session_id):
        self._items[session_id][1] = self.clock()
        self._items.move_to_end(session_id)
        return self._items[session_id][0]

    def __getitem__(self, session_id):
        with self._lock:
            self._expire()
            if session_id not in self._items:
                raise KeyError(session_id)
            return self._touch(session_id)

    def __contains__(self, session_id):
        with self._lock:
            self._expire()
            return session_id in self._items

    def __len__(self):
        with self._lock:
            self._expire()
            return len(self._items)

    def pop(self, session_id, default=None):
        with self._lock:
            item = self._items.pop(session_id, None)
            return item[0] if item else default


SESSIONS = SessionStore()

EVIDENCE_PHRASES = [
    (r"(?:systematic\s+)?reviews?|meta[- ]analys[ie]s|meta[- ]analyses", "systematic_review_meta_analysis"),
    (r"(?:randomi[sz]ed\s+(?:controlled\s+)?)?trials?|rcts?", "randomized_controlled_trial"),
    (r"case\s+reports?|case\s+series", "case_report_series"),
    (r"cohort\s+stud(?:y|ies)", "observational"),
]
EXCLUSION_RE = re.compile(
    r"(?:(?<![a-z])(?:except|without|excluding)|(?:^|[,;]\s*|\b(?:and|but)\s+)not)\s+([a-z0-9][a-z0-9-]*)",
    re.IGNORECASE,
)
CUES = ["what about", "how about", "and", "also", "only", "except", "without"]
PRONOUNS = {"it", "they", "them", "this", "these", "those"}
FOLLOWUP_IGNORE = re.compile(
    r"\b(?:work|works|working|help|helps|know|known|happen|happens|get|make|tell|show|find)\b", re.IGNORECASE
)


def new_session():
    return {"turns": [], "concepts": [], "boolean_query": "", "evidence_filter": None, "clarifications": []}


def extract_evidence_filter(text):
    """Returns (class id or None, text without the phrase)."""
    for phrase, class_id in EVIDENCE_PHRASES:
        pattern = re.compile(rf"\b(?:only|just)\s+(?:the\s+)?(?:{phrase})\b|\b(?:{phrase})\s+only\b", re.IGNORECASE)
        if pattern.search(text):
            return class_id, pattern.sub(" ", text)
    return None, text


def extract_exclusions(text, index, config):
    """Returns (excluded concepts, text without the 'not X' parts). X is one word or lexicon group."""
    excluded = []
    for match in EXCLUSION_RE.finditer(text):
        found = understand(match.group(1), index, config)
        if found:
            excluded.append(dict(found[0], excluded=True, eliminated=False))
    return excluded, EXCLUSION_RE.sub(" ", text)


def followup_signal(text):
    """Returns a short description of the leading cue or pronoun in the text, or None."""
    lowered = text.lower().strip()
    for cue in CUES:
        if re.match(rf"{cue}\b", lowered):
            return f"leading cue '{cue}'"
    pronouns = PRONOUNS & set(re.findall(r"[a-z]+", lowered))
    return f"pronoun '{sorted(pronouns)[0]}'" if pronouns else None


def jaccard(left_labels, right_labels):
    union = left_labels | right_labels
    return len(left_labels & right_labels) / len(union) if union else 1.0


def positive_labels(concepts):
    return {c["label"] for c in concepts if not c.get("excluded")}


def merge_followup(session_concepts, new_concepts, excluded, rules):
    """Adds the new concepts to the session concepts; a concept with the same facet replaces the old one."""
    merged = [dict(c) for c in session_concepts]
    for concept in new_concepts:
        facet = concept.get("facet")
        same_facet = next((i for i, m in enumerate(merged) if facet and m.get("facet") == facet and not m.get("excluded")), None)
        if same_facet is not None:
            rules.append({"rule": "FACET_SWAP", "detail": f"{facet}: {merged[same_facet]['label']} -> {concept['label']}"})
            merged[same_facet] = concept
        elif concept["label"] not in {m["label"] for m in merged}:
            merged.append(concept)
    for concept in excluded:
        merged = [m for m in merged if m["label"] != concept["label"]]
        merged.append(concept)
    return merged


def rewritten_query(concepts, evidence_filter):
    parts = [c["label"] for c in concepts if not c.get("excluded")]
    text = " AND ".join(parts)
    for concept in concepts:
        if concept.get("excluded"):
            text += f" AND NOT {concept['label']}"
    return text + (f" [only: {evidence_filter}]" if evidence_filter else "")


def update_concepts(session, text, original_text, excluded, index, config, rules):
    """Applies FOLLOWUP_DETECT, FACET_SWAP and TOPIC_SWITCH. Returns the new concept list."""
    if not session["concepts"]:
        return understand(text, index, config) + excluded
    new_concepts = understand(FOLLOWUP_IGNORE.sub(" ", text), index, config)
    signal = followup_signal(original_text)
    overlap = jaccard({c["label"] for c in new_concepts}, positive_labels(session["concepts"]))
    if signal is None and len(new_concepts) >= 2 and overlap < config["topic_switch_jaccard"]:
        rules.append({"rule": "TOPIC_SWITCH", "detail": f"concept overlap {overlap:.2f} is below {config['topic_switch_jaccard']}; new topic"})
        session["evidence_filter"] = None
        return new_concepts + excluded
    if signal is None and len(new_concepts) > 2:
        signal = f"{len(new_concepts)} concepts with overlap {overlap:.2f}"
    rules.append({"rule": "FOLLOWUP_DETECT", "detail": signal or "at most 2 concepts"})
    return merge_followup(session["concepts"], new_concepts, excluded, rules)


def bm25_full_collection(concepts, session, index, config):
    """
    The "bm25" ranker: BM25 over the whole collection on the concept stems. There is no Boolean set; the evidence
    filter still applies, and documents containing an excluded concept are removed from the ranking.
    Returns (ranked, trace, clarifications, ranking stems, hits).
    """
    stems = sorted({s for c in concepts if not c.get("excluded") for s in c["stems"]})
    excluded_stems = {s for c in concepts if c.get("excluded") for s in c["stems"]}
    allowed = docs_of_class(index, session["evidence_filter"]) if session["evidence_filter"] else None
    excluded_docs = {d for s in excluded_stems for d, _ in index.get_postings(s)}
    matching = {d for s in stems for d, _ in index.get_postings(s)} - excluded_docs
    matching = matching if allowed is None else matching & allowed
    weight, top_k = config["evidence_weight"], config["top_k"]
    depth = BM25_DEPTH if excluded_docs else (max(config["evidence_candidates"], top_k) if weight else top_k)
    ranked = rank_bm25(stems, index, top_k=depth, doc_ids=allowed, k1=config["bm25_k1"], b=config["bm25_b"], title_boost=config["title_boost"])
    ranked = [pair for pair in ranked if pair[1] not in excluded_docs]
    ranked = apply_evidence_weight(ranked, weight, lambda doc_id: g_value(class_of(index, doc_id), config))[:top_k]
    filter_note = f"; evidence filter {session['evidence_filter']} applied" if session["evidence_filter"] else ""
    trace = [{
        "step": 1, "query": "BM25 over the whole collection: " + " ".join(stems),
        "df_table": {s: index.get_df(s) for s in stems}, "estimated_hits": len(matching), "actual_hits": len(matching),
        "decision": "BM25_FULL_COLLECTION",
        "reason": "ranker bm25: no Boolean set is built; every document containing a concept stem is scored" + filter_note,
        "source": "rule",
    }]
    return ranked, trace, [], stems, len(matching)


def concept_status(concept):
    """How a concept took part in the last run: active, left_out (held back or too common, never added), dropped or excluded."""
    if concept.get("excluded"):
        return "excluded"
    if concept.get("dropped"):
        return "dropped"
    if concept.get("held") or concept["eliminated"]:
        return "left_out"
    return "active"


def agent_turn(text, concepts, session, index, config, allow_clarify):
    ranked, log, boolean_query = run_controller(
        text, index, config, concepts=concepts, allow_clarify=allow_clarify, evidence_filter=session["evidence_filter"]
    )
    statuses = {c["label"]: concept_status(c) for c in log.concepts}
    return ranked, log.steps, log.clarifications, log.ranking_stems, len(log.final_docs), boolean_query, statuses


def finish_turn(session, text, concepts, index, config, allow_clarify, rules, ranker="agent", record_turn=True):
    mark_elimination(concepts, index, config)
    if ranker == "bm25":
        ranked, trace, clarifications, stems, hits = bm25_full_collection(concepts, session, index, config)
        boolean_query = ""
        statuses = {c["label"]: "excluded" if c.get("excluded") else "active" for c in concepts}
    else:
        ranked, trace, clarifications, stems, hits, boolean_query, statuses = agent_turn(text, concepts, session, index, config, allow_clarify)
    session["concepts"] = concepts
    session["boolean_query"] = boolean_query
    session["ranking_stems"] = stems
    session["clarifications"] = clarifications
    result = {
        "rewritten_query": rewritten_query(concepts, session["evidence_filter"]),
        "rules_fired": rules,
        "boolean_query": boolean_query,
        "trace": trace,
        "clarifications": clarifications,
        "results": [
            {
                "doc_id": doc_id, "ext_id": index.get_ext_id(doc_id), "title": index.get_doc_title(doc_id),
                "score": score, "evidence_class": class_of(index, doc_id),
            }
            for score, doc_id in ranked
        ],
        "evidence_filter": session["evidence_filter"],
        "concepts": concepts,
        "concept_statuses": statuses,
        "ranking_stems": stems,
        "hits": hits,
        "ranker": ranker,
    }
    if record_turn:
        session["turns"].append({"text": text, "rewritten_query": result["rewritten_query"], "rules_fired": rules})
    return result


def rerank_turn(session_id, ranker, index, config):
    """Ranks the current turn again with the chosen ranker, in place: no new turn, same concepts, filter and rules."""
    session = SESSIONS[session_id]
    last = session["turns"][-1]
    concepts = [dict(c, stems=list(c["stems"])) for c in session["concepts"]]
    return finish_turn(session, last["text"], concepts, index, config, ranker == "agent", last["rules_fired"], ranker, record_turn=False)


def handle_turn(session_id, text, index, config, allow_clarify=True, ranker="agent", evidence_filter=KEEP_FILTER):
    """
    One conversation turn. ranker is "agent" (Boolean controller) or "bm25" (full-collection BM25).
    evidence_filter KEEP_FILTER keeps the session's filter; a class id sets it and None clears it.
    """
    session = SESSIONS.setdefault(session_id, new_session())
    if evidence_filter is not KEEP_FILTER:
        session["evidence_filter"] = validate_class(evidence_filter) if evidence_filter else None
    rules = []
    evidence_class, remaining = extract_evidence_filter(text)
    excluded, remaining = extract_exclusions(remaining, index, config)
    concepts = update_concepts(session, remaining, text, excluded, index, config, rules)
    if evidence_class:
        session["evidence_filter"] = evidence_class
        rules.insert(0, {"rule": "EVIDENCE_FILTER", "detail": evidence_class})
    if excluded:
        rules.insert(1 if evidence_class else 0, {"rule": "EXCLUSION", "detail": ", ".join(c["label"] for c in excluded)})
    return finish_turn(session, text, concepts, index, config, allow_clarify, rules, ranker)


def apply_clarification(session_id, choice_id, index, config):
    """Adds the chosen cluster's label stems as a required concept and reruns the controller."""
    session = SESSIONS[session_id]
    choice = next(c for c in session["clarifications"] if c["id"] == choice_id)
    stems = [stem for stem in choice["label_stems"] if index.get_df(stem) > 0]
    concept = make_concept(" ".join(choice["label_terms"]), stems, index, True, "clarify")
    rules = [{"rule": "CLARIFY_CHOSEN", "detail": f"choice {choice_id}: {' '.join(choice['label_terms'])}"}]
    result = finish_turn(session, f"#{choice_id}", session["concepts"] + [concept], index, config, False, rules)
    chosen_row = {
        "step": 0, "query": "", "df_table": {}, "estimated_hits": 0, "actual_hits": 0,
        "decision": "CLARIFY_CHOSEN", "reason": rules[0]["detail"], "source": "rule",
    }
    result["trace"] = [chosen_row] + result["trace"]
    return result
