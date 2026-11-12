"""
conversation.py
Multi-turn sessions. Each turn is rewritten into a full set of concepts by five named rules,
the controller runs on those concepts, and the rewritten query plus the rules that fired are returned.

Rules, in the order they are checked:
  EVIDENCE_FILTER, EXCLUSION, FOLLOWUP_DETECT, FACET_SWAP, TOPIC_SWITCH.
"""

import re

from engine.controller import run_controller
from engine.evidence import class_of
from engine.query_understanding import make_concept, mark_elimination, understand

SESSIONS = {}

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


def finish_turn(session, text, concepts, index, config, allow_clarify, rules):
    mark_elimination(concepts, index, config)
    ranked, log, boolean_query = run_controller(
        text, index, config, concepts=concepts, allow_clarify=allow_clarify, evidence_filter=session["evidence_filter"]
    )
    session["concepts"] = concepts
    session["boolean_query"] = boolean_query
    session["clarifications"] = log.clarifications
    result = {
        "rewritten_query": rewritten_query(concepts, session["evidence_filter"]),
        "rules_fired": rules,
        "boolean_query": boolean_query,
        "trace": log.steps,
        "clarifications": log.clarifications,
        "results": [
            {
                "doc_id": doc_id, "ext_id": index.get_ext_id(doc_id), "title": index.get_doc_title(doc_id),
                "score": score, "evidence_class": class_of(index, doc_id),
            }
            for score, doc_id in ranked
        ],
        "evidence_filter": session["evidence_filter"],
    }
    session["turns"].append({"text": text, "rewritten_query": result["rewritten_query"], "rules_fired": rules})
    return result


def handle_turn(session_id, text, index, config, allow_clarify=True):
    session = SESSIONS.setdefault(session_id, new_session())
    rules = []
    evidence_class, remaining = extract_evidence_filter(text)
    excluded, remaining = extract_exclusions(remaining, index, config)
    concepts = update_concepts(session, remaining, text, excluded, index, config, rules)
    if evidence_class:
        session["evidence_filter"] = evidence_class
        rules.insert(0, {"rule": "EVIDENCE_FILTER", "detail": evidence_class})
    if excluded:
        rules.insert(1 if evidence_class else 0, {"rule": "EXCLUSION", "detail": ", ".join(c["label"] for c in excluded)})
    return finish_turn(session, text, concepts, index, config, allow_clarify, rules)


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
