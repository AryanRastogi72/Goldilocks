"""
evidence.py
The Evidence Lens: a rule-based study-type classifier, the evidence hierarchy value g(d),
and lookups of the labels that `pipeline.build_evidence` saves in index_out/evidence.json.

Rules are checked in order and the first match wins. Title rules come before abstract rules, and within
each field the order is review, trial, case report, model, lab or animal, opinion, observational.
Reviews come first because reviews of trials mention "randomized".
"""

import json
import os
import re

CLASSES = [
    "systematic_review_meta_analysis",
    "randomized_controlled_trial",
    "observational",
    "case_report_series",
    "modelling_simulation",
    "lab_animal",
    "opinion_editorial",
    "other",
]

REVIEW = r"systematic(?:ally)? review|meta-?analys|pooled analysis|scoping review|umbrella review|rapid review|literature review"
TRIAL_TITLE = r"randomi[sz]ed.{0,60}(?:trial|controlled)|(?:trial|controlled).{0,60}randomi[sz]ed|placebo-controlled|double-blind"
TRIAL_ABSTRACT = r"we randomly assigned|were randomly assigned|randomly allocated|were randomi[sz]ed|randomi[sz]ed (?:controlled|clinical|double)"
CASE_TITLE = r"case reports?|\ba case of\b|case series|\ba report of\b"
CASE_ABSTRACT = r"we report (?:a|an|the) (?:patient|case|man|woman|child|infant)|we describe (?:a|an|the) (?:patient|case)|case reports?|case series"
MODEL = r"mathematical model|\bseir\b|\bsir model|compartmental|simulation|forecast"
LAB = r"in vitro|in vivo|\bmice\b|\bmouse\b|murine|cell lines?|macaques?|hamsters?|animal models?|\brats?\b"
OPINION_TITLE = r"editorial|commentary|letter to the editor|perspective|viewpoint|correspondence"
OPINION_ABSTRACT = r"in this (?:editorial|commentary|perspective|viewpoint|letter)|this (?:editorial|commentary|letter)"
OBSERVATIONAL = r"cohort|case-control|cross-sectional|retrospective|prospective|registry"

RULES = [
    {"id": "T_REVIEW", "class": "systematic_review_meta_analysis", "field": "title", "pattern": REVIEW},
    {"id": "T_RCT", "class": "randomized_controlled_trial", "field": "title", "pattern": TRIAL_TITLE},
    {"id": "T_CASE", "class": "case_report_series", "field": "title", "pattern": CASE_TITLE},
    {"id": "T_MODEL", "class": "modelling_simulation", "field": "title", "pattern": MODEL},
    {"id": "T_LAB", "class": "lab_animal", "field": "title", "pattern": LAB},
    {"id": "T_OPINION", "class": "opinion_editorial", "field": "title", "pattern": OPINION_TITLE},
    {"id": "T_OBS", "class": "observational", "field": "title", "pattern": OBSERVATIONAL},
    {"id": "A_REVIEW", "class": "systematic_review_meta_analysis", "field": "abstract", "pattern": REVIEW},
    {"id": "A_RCT", "class": "randomized_controlled_trial", "field": "abstract", "pattern": TRIAL_ABSTRACT},
    {"id": "A_CASE", "class": "case_report_series", "field": "abstract", "pattern": CASE_ABSTRACT},
    {"id": "A_MODEL", "class": "modelling_simulation", "field": "abstract", "pattern": MODEL},
    {"id": "A_LAB", "class": "lab_animal", "field": "abstract", "pattern": LAB},
    {"id": "A_OPINION", "class": "opinion_editorial", "field": "abstract", "pattern": OPINION_ABSTRACT},
    {"id": "A_OBS", "class": "observational", "field": "abstract", "pattern": OBSERVATIONAL},
]
NO_MATCH = "NO_MATCH"
COMPILED_RULES = [dict(rule, regex=re.compile(rule["pattern"])) for rule in RULES]

_labels_cache = {}


def classify(title, abstract):
    """Returns (class, rule_id, matched_text). The first matching rule wins."""
    text = {"title": (title or "").lower(), "abstract": (abstract or "").lower()}
    for rule in COMPILED_RULES:
        match = rule["regex"].search(text[rule["field"]])
        if match:
            return rule["class"], rule["id"], match.group(0)
    return "other", NO_MATCH, ""


def validate_class(class_id):
    """Raises ValueError for a class id that is not one of the evidence classes."""
    if class_id not in CLASSES:
        raise ValueError(f"unknown evidence class '{class_id}'; expected one of {CLASSES}")
    return class_id


def g_value(class_id, config):
    """Evidence hierarchy value in [0, 1] of a class (config evidence_g)."""
    return config["evidence_g"][validate_class(class_id)]


def classify_doc(index, doc_id):
    title, abstract = index.get_title_and_abstract(doc_id)
    return classify(title, abstract)


def load_labels(index_dir):
    """Reads index_out/evidence.json once: {"classes", "rules", "labels": [[class_idx, rule_idx], ...]}."""
    if index_dir not in _labels_cache:
        with open(os.path.join(index_dir, "evidence.json"), "r") as f:
            _labels_cache[index_dir] = json.load(f)
    return _labels_cache[index_dir]


def class_of(index, doc_id):
    data = load_labels(index.index_dir)
    return data["classes"][data["labels"][doc_id][0]]


def docs_of_class(index, class_id):
    """Set of doc ids labelled with the class."""
    validate_class(class_id)
    data = load_labels(index.index_dir)
    class_idx = data["classes"].index(class_id)
    return {doc_id for doc_id, (idx, _) in enumerate(data["labels"]) if idx == class_idx}
