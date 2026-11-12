import contextlib
import html
import io
import json
import math
import re
import time

import pandas as pd
import streamlit as st

from engine.controller import run_controller
from pipeline.tokenizer import stem_term, tokenize
from ui_styles import inject_styles


PALETTE = ["#6C5CE7", "#14B8A6", "#F59E0B", "#3B82F6"]


def safe(value):
    return html.escape(str(value), quote=True)


def get_config(root):
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def render_sidebar(config):
    with st.sidebar:
        st.markdown("## Search controls")
        st.number_input(
            "Target minimum",
            min_value=1,
            value=int(config.get("target_band_low", 20)),
            key="target_low",
        )
        st.number_input(
            "Target maximum",
            min_value=1,
            value=int(config.get("target_band_high", 200)),
            key="target_high",
        )
        st.number_input(
            "Iteration limit",
            min_value=1,
            max_value=20,
            value=int(config.get("max_iterations", 8)),
            key="iteration_limit",
        )
        st.number_input(
            "Top K papers",
            min_value=1,
            max_value=100,
            value=int(config.get("top_k", 10)),
            key="top_k",
        )
        st.checkbox("Skip step animation", key="skip_animation", value=False)


def render_header(index):
    st.markdown(
        '<div class="brand-row"><div class="brand-mark">G</div>'
        '<div class="brand-name">Goldilocks</div></div>'
        '<div class="brand-tagline">Search results that are just the right size</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="status-row">'
        f'<span class="status-chip"><span class="status-dot good"></span>'
        f'Index loaded, {index.num_docs:,} documents</span>'
        f'<span class="status-chip"><span class="status-dot brand"></span>'
        f'Offline rules, no model API</span></div>',
        unsafe_allow_html=True,
    )


def active_settings(config):
    low = int(st.session_state.get("target_low", config.get("target_band_low", 20)))
    high = int(st.session_state.get("target_high", config.get("target_band_high", 200)))
    if low > high:
        return None
    result = config.copy()
    result.update(
        {
            "target_band_low": low,
            "target_band_high": high,
            "max_iterations": int(st.session_state.get("iteration_limit", 8)),
            "top_k": int(st.session_state.get("top_k", 10)),
        }
    )
    return result


def run_search(query, index, config):
    st.session_state.pop("search_result", None)
    started = time.perf_counter()

    with st.status("Searching the indexed collection", expanded=False) as status:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ranked, trace, final_query = run_controller(query, index, config)
        except Exception:
            status.update(label="Search could not finish", state="error")
            st.error("Search could not finish. Check the query and index, then try again.")
            return None

        elapsed = time.perf_counter() - started
        status.update(label="Search complete", state="complete")

    st.session_state["search_result"] = {
        "query": query,
        "ranked": ranked,
        "trace": trace,
        "final_query": final_query,
        "settings": config.copy(),
        "elapsed": elapsed,
    }
    return st.session_state["search_result"]


def log_position(value, total):
    ceiling = max(int(total), 2)
    return max(0.0, min(100.0, math.log10(max(int(value), 1)) / math.log10(ceiling) * 100))


def render_gauge(hit_count, total, low, high):
    target_start = log_position(low, total)
    target_end = log_position(high, total)
    marker = log_position(hit_count, total)
    gauge = (
        '<div class="section-title">Result size</div>'
        '<p class="section-copy">The scale is logarithmic, so common and rare result sizes stay visible together.</p>'
        f'<div class="gauge-topline"><span>1 document</span>'
        f'<span>Current result, <strong>{hit_count:,}</strong></span>'
        f'<span>{total:,} documents</span></div>'
        f'<div class="gauge-track" role="img" aria-label="{hit_count} hits on a log scale from 1 to {total}, target {low} to {high}" '
        f'style="--target-start:{target_start:.3f}%;--target-end:{target_end:.3f}%;--marker:{marker:.3f}%">'
        '<div class="gauge-marker"></div></div>'
        f'<div class="gauge-zone-labels" style="--target-start:{target_start:.3f}%;--target-end:{target_end:.3f}%">'
        '<span>Too few</span><span>Just right</span><span>Too many</span></div>'
    )
    st.markdown(
        gauge,
        unsafe_allow_html=True,
    )


def decision_class(decision):
    if decision == "ACCEPT":
        return "decision-accept", "✓"
    if decision.startswith("TIGHTEN"):
        return "decision-tighten", "+"
    if decision.startswith("RELAX"):
        return "decision-relax", "↗"
    if decision == "STOP":
        return "decision-stop", "×"
    return "decision-info", "i"


def step_terms(row, total):
    terms = [term for term in tokenize(row.get("query", "")) if term not in {"and", "or", "not"}]
    df_table = row.get("df_table", {})
    chips = []
    for term in dict.fromkeys(terms):
        df = int(df_table.get(term, 0))
        width = log_position(df, total) if df else 0
        chips.append(
            f'<div class="term-chip"><div class="term-chip-top">'
            f'<span>{safe(term)}</span><span>df {df:,}</span></div>'
            f'<div class="df-track"><div class="df-fill" style="width:{width:.2f}%"></div></div></div>'
        )
    if not chips:
        return ""
    return f'<div class="term-chip-list">{"".join(chips)}</div>'


def postings_rows(index, query):
    terms = [term for term in tokenize(query) if term not in {"and", "or", "not"}]
    rows = []
    for term in dict.fromkeys(terms):
        for doc_id, tf in index.get_postings(term)[:10]:
            rows.append(
                {
                    "Term": term,
                    "Document ID": index.get_ext_id(doc_id),
                    "TF": int(tf),
                }
            )
    return rows


def render_step(row, index, animate):
    decision = str(row.get("decision", ""))
    badge_class, icon = decision_class(decision)
    animation_class = "animate-step" if animate else ""
    delay = min(int(row.get("step", 1)) * 55, 440)
    st.markdown(
        f'<div class="step-head {animation_class}" style="animation-delay:{delay}ms">'
        f'<div class="step-number">{int(row.get("step", 0)):02d}</div>'
        f'<div><div class="step-query">{safe(row.get("query", ""))}</div>'
        f'<span class="decision-badge {badge_class}"><span class="decision-icon" aria-hidden="true">{icon}</span>'
        f'{safe(decision)}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="step-counts"><span class="count-pill">Estimate {int(row.get("estimated_hits", 0)):,}</span>'
        f'<span class="count-pill">Actual hits {int(row.get("actual_hits", 0)):,}</span></div>'
        f'<div class="step-reason">{safe(row.get("reason", ""))}</div>'
        f'{step_terms(row, index.num_docs)}',
        unsafe_allow_html=True,
    )
    with st.expander(f"Postings preview for step {int(row.get('step', 0))}"):
        rows = postings_rows(index, row.get("query", ""))
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No postings are available for the terms in this step.")


def matched_title(title, query_terms):
    wanted = set(query_terms)

    def mark(match):
        word = match.group(0)
        normalized = stem_term(word.lower())
        escaped = safe(word)
        return f"<mark>{escaped}</mark>" if normalized in wanted else escaped

    return re.sub(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", mark, str(title))


def render_results(result, index):
    st.markdown('<div class="section-title">Ranked papers</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Boolean retrieval finds the candidates. BM25 orders them.</p>',
        unsafe_allow_html=True,
    )
    ranked = result["ranked"]
    if not ranked:
        st.info("The final Boolean query returned no papers.")
        return
    query_terms = [term for term in tokenize(result["final_query"]) if term not in {"and", "or", "not"}]
    max_score = max((float(score) for score, _ in ranked), default=0)
    for rank, (score, doc_id) in enumerate(ranked, start=1):
        title = index.get_doc_title(doc_id)
        external_id = index.get_ext_id(doc_id)
        with st.container(border=True):
            left, right = st.columns([0.78, 0.22], vertical_alignment="center")
            with left:
                st.markdown(f'<div class="result-title">{safe(title)}</div>', unsafe_allow_html=True)
            with right:
                st.markdown(f'<div class="mono" style="text-align:right;color:#596579">Rank {rank}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="title-preview">Title preview, '
                f'{matched_title(title, query_terms)}</div>',
                unsafe_allow_html=True,
            )
            score_ratio = max(0.0, min(1.0, float(score) / max_score)) if max_score > 0 else 0.0
            st.progress(score_ratio, text=f"BM25 score {float(score):.4f}")
            st.markdown(
                f'<div class="result-meta"><span>Document ID <span class="mono">{safe(external_id)}</span></span>'
                f'<span>Score {float(score):.4f}</span></div>',
                unsafe_allow_html=True,
            )


def render_search(root, index, config):
    current_config = active_settings(config)
    if current_config is None:
        st.error("The target minimum must be no greater than the target maximum.")
        return

    examples = ["COVID vaccine pregnancy", "masks", "what is the origin of covid-19"]
    st.markdown('<div class="section-title">Start with a research question</div>', unsafe_allow_html=True)
    example_cols = st.columns(3)
    submitted_by_example = False
    for position, example in enumerate(examples):
        if example_cols[position].button(example, key=f"example_{position}", use_container_width=True):
            st.session_state["search_query"] = example
            submitted_by_example = True

    with st.form("search_form"):
        query_col, search_col = st.columns([0.82, 0.18], vertical_alignment="bottom")
        with query_col:
            query = st.text_input(
                "Research question",
                key="search_query",
                placeholder="For example, long term symptoms after COVID",
            )
        with search_col:
            submitted = st.form_submit_button("Search", type="primary", use_container_width=True)
    submitted = submitted or submitted_by_example

    if submitted:
        if not query.strip():
            st.warning("Enter a research question to search the collection.")
        else:
            run_search(query.strip(), index, current_config)

    result = st.session_state.get("search_result")
    if result is None:
        st.markdown(
            '<div class="empty-state"><div class="empty-title">Your search trace will appear here</div>'
            '<div>Choose an example or enter a question to inspect the real postings, document frequencies, '
            "controller decisions, and ranked papers.</div></div>",
            unsafe_allow_html=True,
        )
        return

    rows = result["trace"].steps
    final_hits = rows[-1]["actual_hits"] if rows else 0
    settings = result["settings"]
    render_gauge(final_hits, index.num_docs, settings["target_band_low"], settings["target_band_high"])

    metrics = st.columns(3)
    metrics[0].metric("Final hits", f"{final_hits:,}")
    metrics[1].metric("Steps used", len(rows))
    elapsed = result["elapsed"]
    metrics[2].metric("Elapsed time", f"{elapsed * 1000:.0f} ms" if elapsed < 1 else f"{elapsed:.2f} sec")

    st.markdown('<div class="section-title">Controller timeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Every decision name is the engine trace name. Estimates and actual hits come from the index.</p>',
        unsafe_allow_html=True,
    )
    animate = not st.session_state.get("skip_animation", False)
    for row in rows:
        with st.container(border=True):
            render_step(row, index, animate)

    st.markdown('<div class="section-title">Final Boolean query</div>', unsafe_allow_html=True)
    st.code(result["final_query"], language=None)
    render_results(result, index)


def render_pipeline():
    steps = [
        ("Corpus", "Collection documents and metadata"),
        ("Tokenizer and stemmer", "Token normalization"),
        ("Inverted index", "Dictionary and postings lists"),
        ("Dictionary with df", "Document frequency statistics"),
        ("Boolean engine", "AND, OR, and NOT set operations"),
        ("Controller", "Query processing and reformulation"),
        ("BM25 ranker", "Probabilistic term weighting"),
    ]
    cards = []
    for name, concept in steps:
        cards.append(
            f'<div class="pipeline-card"><div class="pipeline-name">{safe(name)}</div>'
            f'<div class="pipeline-concept">Lecture concept: {safe(concept)}</div></div>'
        )
    st.markdown(
        '<div class="pipeline-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def render_how_it_works(config):
    st.markdown('<div class="section-title">From collection to ranked papers</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Each card links a real project component to an information retrieval lecture topic.</p>',
        unsafe_allow_html=True,
    )
    render_pipeline()
    st.markdown('<div class="section-title">Controller rule</div>', unsafe_allow_html=True)
    rules = pd.DataFrame(
        [
            {"Observed actual hits": f"Above {config['target_band_high']}", "Action": "Tighten the query"},
            {"Observed actual hits": f"Below {config['target_band_low']}", "Action": "Relax the query"},
            {"Observed actual hits": f"{config['target_band_low']} through {config['target_band_high']}", "Action": "Accept the query"},
            {"Observed actual hits": "Iteration limit reached", "Action": "Stop adjusting"},
        ]
    )
    st.dataframe(rules, hide_index=True, use_container_width=True)
    st.caption("The estimated bound comes from document frequency. The controller checks the actual Boolean hit count before it accepts, tightens, or relaxes.")


def render_app(root, load_index):
    inject_styles()
    config = get_config(root)
    render_sidebar(config)
    with st.status("Loading the collection index", expanded=False) as status:
        try:
            index = load_index(str(root / "index_out"))
            status.update(label=f"Index ready, {index.num_docs:,} documents", state="complete")
        except FileNotFoundError:
            status.update(label="Index files are missing", state="error")
            st.error("The collection index is missing. Build the index before starting search.")
            return

    render_header(index)
    search_tab, guide_tab = st.tabs(["Search", "How it works"])
    with search_tab:
        render_search(root, index, config)
    with guide_tab:
        render_how_it_works(config)
