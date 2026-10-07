import contextlib
import html
import io
import json
import math
import os
import re
import time
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from engine.controller import run_controller
from engine.llm_client import using_api_key
from engine.ranker import compute_query_vector
from pipeline.tokenizer import stem_term, tokenize
from ui_styles import inject_styles


PALETTE = ["#6C5CE7", "#14B8A6", "#F59E0B", "#3B82F6"]


def safe(value):
    return html.escape(str(value), quote=True)


def get_config(root):
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def owner_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


def reset_key_status():
    st.session_state["key_rejected"] = False


def render_sidebar(config):
    with st.sidebar:
        st.markdown("## Search controls")
        st.radio(
            "Query proposal mode",
            ["Offline", "Groq"],
            horizontal=True,
            key="proposal_mode",
        )
        st.text_input(
            "Groq API key",
            type="password",
            key="session_api_key",
            on_change=reset_key_status,
            help="The project currently sends model requests to Groq.",
        )
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
        st.markdown(
            '<div class="privacy-note">A key typed here stays in this browser session. '
            "The app owner key, when configured, stays in app secrets."
            "</div>",
            unsafe_allow_html=True,
        )
        st.checkbox("Skip step animation", key="skip_animation", value=False)


def effective_key():
    return st.session_state.get("session_api_key", "").strip() or owner_key()


def key_status():
    if st.session_state.get("key_rejected", False):
        return "Key rejected", "bad"
    if effective_key():
        return "Key set", "good"
    return "Key not set", "warn"


def render_header(index):
    mode = st.session_state.get("proposal_mode", "Offline")
    key_label, key_class = key_status()
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
        f'Mode, {safe(mode)}</span>'
        f'<span class="status-chip"><span class="status-dot {key_class}"></span>'
        f'{safe(key_label)}</span></div>',
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


def run_search(query, index, config, context="", context_query=None):
    st.session_state.pop("search_result", None)
    mode = st.session_state.get("proposal_mode", "Offline")
    api_key = effective_key()
    use_model = mode == "Groq"
    error_message = ""
    started = time.perf_counter()

    with st.status("Searching the indexed collection", expanded=False) as status:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if use_model:
                    with using_api_key(api_key):
                        ranked, trace, final_query = run_controller(
                            query,
                            index,
                            config,
                            use_llm=True,
                            context=context,
                            context_query=context_query,
                        )
                    if trace.llm_errors:
                        message = " ".join(str(item) for item in trace.llm_errors)
                        if "401" in message or "403" in message:
                            st.session_state["key_rejected"] = True
                            error_message = "Groq rejected this key. The search used offline keywords."
                        elif "429" in message or "rate" in message.lower():
                            error_message = "Groq rate limit reached. The search used offline keywords."
                        elif "not configured" in message:
                            error_message = "No Groq key is set and no cached proposal matched. The search used offline keywords."
                        elif "contained no text" in message:
                            error_message = "Groq returned no query text. The search used offline keywords."
                        elif any(f"HTTP {code}" in message for code in (500, 502, 503, 504)) or message == "Groq request failed. Check the connection and model settings.":
                            error_message = "Groq had a temporary service or connection problem. The search used offline keywords."
                        else:
                            error_message = "Groq was unavailable. The search used offline keywords."
                        ranked, trace, final_query = run_controller(
                            query,
                            index,
                            config,
                            use_llm=False,
                            context_query=context_query,
                        )
                        mode_used = "Offline"
                    else:
                        mode_used = "Groq"
                else:
                    ranked, trace, final_query = run_controller(
                        query,
                        index,
                        config,
                        use_llm=False,
                        context_query=context_query,
                    )
                    mode_used = "Offline"
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
        "mode_used": mode_used,
        "settings": config.copy(),
        "elapsed": elapsed,
        "notice": error_message,
        "context_used": bool(context_query),
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
    delay = min(int(row.get("iteration", 1)) * 55, 440)
    st.markdown(
        f'<div class="step-head {animation_class}" style="animation-delay:{delay}ms">'
        f'<div class="step-number">{int(row.get("iteration", 0)):02d}</div>'
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
    with st.expander(f"Postings preview for step {int(row.get('iteration', 0))}"):
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


def document_term_stats(index, doc_ids, query_terms):
    query_weights = compute_query_vector(query_terms, index)
    query_norm = math.sqrt(sum(weight**2 for weight in query_weights.values()))
    wanted_docs = set(doc_ids)
    output = {doc_id: [] for doc_id in wanted_docs}
    for term, query_weight in query_weights.items():
        idf = index.get_idf(term)
        for doc_id, tf in index.get_postings(term):
            if doc_id not in wanted_docs:
                continue
            doc_weight = (1 + math.log10(tf)) * idf
            contribution = query_weight * doc_weight
            normalized = contribution / (index.get_doc_norm(doc_id) * query_norm) if query_norm else 0
            output[doc_id].append(
                {
                    "Term": term,
                    "Query TF": query_terms.count(term),
                    "Document TF": int(tf),
                    "IDF": round(idf, 5),
                    "Document weight": round(doc_weight, 5),
                    "Score contribution": round(normalized, 6),
                }
            )
    return output


def render_results(result, index):
    st.markdown('<div class="section-title">Ranked papers</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Boolean retrieval finds the candidates. TF IDF cosine similarity orders them.</p>',
        unsafe_allow_html=True,
    )
    ranked = result["ranked"]
    if not ranked:
        st.info("The final Boolean query returned no papers.")
        return
    query_terms = [term for term in tokenize(result["final_query"]) if term not in {"and", "or", "not"}]
    max_score = max((float(score) for score, _ in ranked), default=0)
    term_stats = document_term_stats(
        index,
        [doc_id for _, doc_id in ranked],
        query_terms,
    )
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
            st.progress(score_ratio, text=f"TF IDF cosine score {float(score):.6f}")
            st.markdown(
                f'<div class="result-meta"><span>Document ID <span class="mono">{safe(external_id)}</span></span>'
                f'<span>Score {float(score):.6f}</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Why this score"):
                stats = term_stats.get(doc_id, [])
                if stats:
                    st.dataframe(pd.DataFrame(stats), hide_index=True, use_container_width=True)
                    st.caption("Term contributions are normalized by the query and document vector lengths.")
                else:
                    st.info("No query term contributes to this document score.")


def render_search(root, index, config):
    current_config = active_settings(config)
    if current_config is None:
        st.error("The target minimum must be no greater than the target maximum.")
        return

    history = st.session_state.setdefault("conversation_history", [])
    if history:
        with st.container(border=True):
            st.markdown("### Recent conversation")
            for turn in history[-3:]:
                st.caption(f"Question: {turn['question']}")
                st.caption(f"Last accepted query: {turn['final_query']}")
            use_prior_context = st.checkbox(
                "Use earlier searches as context",
                value=True,
                key=f"use_prior_context_{len(history)}",
            )
        if st.button("New conversation", key="new_conversation"):
            st.session_state["conversation_history"] = []
            st.session_state.pop("search_result", None)
            st.session_state["search_query"] = ""
            st.rerun()
    else:
        use_prior_context = False

    examples = ["COVID vaccine pregnancy", "COVID remdesivir pregnant treatment", "long COVID symptoms"]
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
            context = ""
            context_query = None
            if use_prior_context and history:
                context = "\n".join(
                    f"Question: {turn['question']}\nAccepted Boolean query: {turn['final_query']}"
                    for turn in history[-3:]
                )
                context_query = history[-1]["final_query"]
            completed = run_search(
                query.strip(),
                index,
                current_config,
                context=context,
                context_query=context_query,
            )
            if completed:
                history.append(
                    {
                        "question": query.strip(),
                        "final_query": completed["final_query"],
                        "mode": completed["mode_used"],
                    }
                )

    result = st.session_state.get("search_result")
    if result is None:
        st.markdown(
            '<div class="empty-state"><div class="empty-title">Your search trace will appear here</div>'
            '<div>Choose an example or enter a question to inspect the real postings, document frequencies, '
            "controller decisions, and ranked papers.</div></div>",
            unsafe_allow_html=True,
        )
        return

    if result.get("notice"):
        st.warning(result["notice"])

    rows = result["trace"].iterations
    final_hits = rows[-1]["actual_hits"] if rows else 0
    settings = result["settings"]
    render_gauge(final_hits, index.num_docs, settings["target_band_low"], settings["target_band_high"])

    metrics = st.columns(4)
    metrics[0].metric("Final hits", f"{final_hits:,}")
    metrics[1].metric("Iterations used", len(rows))
    metrics[2].metric("LLM calls", result["trace"].llm_calls)
    elapsed = result["elapsed"]
    metrics[3].metric("Elapsed time", f"{elapsed * 1000:.0f} ms" if elapsed < 1 else f"{elapsed:.2f} sec")

    st.markdown('<div class="section-title">Controller timeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Every decision name is the engine trace name. Estimates and actual hits come from the index.</p>',
        unsafe_allow_html=True,
    )
    animate = not st.session_state.get("skip_animation", False)
    for row in rows:
        with st.container(border=True):
            render_step(row, index, animate)

    st.markdown('<div class="section-title">Accepted Boolean query</div>', unsafe_allow_html=True)
    st.code(result["final_query"], language=None)
    st.caption("Use the copy control on the query box to copy this query.")
    render_results(result, index)
    st.caption("The index stores document titles and metadata, not article text. Highlighted previews therefore use the real title only.")


def evaluation_rows(root):
    path = root / "evaluation" / "results.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    systems = [
        ("Goldilocks", "goldilocks"),
        ("AND baseline", "and_baseline"),
        ("OR baseline", "or_baseline"),
        ("One shot Groq baseline", "one_shot_llm_baseline"),
    ]
    summaries = []
    for label, key in systems:
        entries = [item for item in data.get(key, []) if item.get("available", True)]
        if not entries:
            summaries.append({"System": label, "Available": False})
            continue
        count = len(entries)
        in_band = sum(bool(item["in_band"]) for item in entries)
        summaries.append(
            {
                "System": label,
                "Available": True,
                "Topics": count,
                "In band rate": in_band / count,
                "Average hits": sum(item["hits"] for item in entries) / count,
                "Precision": sum(item["precision"] for item in entries) / count,
                "Recall": sum(item["recall"] for item in entries) / count,
                "P at 10": sum(item["p_at_10"] for item in entries) / count,
                "nDCG at 10": sum(item["ndcg_at_10"] for item in entries) / count,
                "MRR at 10": sum(item["mrr_at_10"] for item in entries) / count,
            }
        )
    return data, summaries


def render_evaluation(root):
    st.markdown('<div class="section-title">Saved evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">All values below are calculated from the saved topic results and relevance judgments.</p>',
        unsafe_allow_html=True,
    )
    data, summaries = evaluation_rows(root)
    if data is None:
        st.info("No saved evaluation output is available.")
        return

    card_columns = st.columns(2)
    for position, summary in enumerate(summaries):
        with card_columns[position % 2]:
            with st.container(border=True):
                st.markdown(f"### {summary['System']}")
                if not summary["Available"]:
                    st.info("Not scored in the saved evaluation.")
                    continue
                values = [
                    ("In band rate", f"{summary['In band rate']:.1%}"),
                    ("Average hits", f"{summary['Average hits']:,.1f}"),
                    ("Topics", str(summary["Topics"])),
                    ("Precision", f"{summary['Precision']:.4f}"),
                    ("Recall", f"{summary['Recall']:.4f}"),
                    ("P at 10", f"{summary['P at 10']:.4f}"),
                    ("nDCG at 10", f"{summary['nDCG at 10']:.4f}"),
                    ("MRR at 10", f"{summary['MRR at 10']:.4f}"),
                ]
                value_html = "".join(
                    f'<div class="eval-item"><div class="eval-label">{safe(label)}</div>'
                    f'<div class="eval-value">{safe(value)}</div></div>'
                    for label, value in values
                )
                st.markdown(f'<div class="eval-grid">{value_html}</div>', unsafe_allow_html=True)

    one_shot_entries = data.get("one_shot_llm_baseline", [])
    one_shot_available = [entry for entry in one_shot_entries if entry.get("available", True)]
    one_shot_unavailable = [entry for entry in one_shot_entries if not entry.get("available", True)]
    no_proposal_count = sum("not configured" in entry.get("error", "").lower() for entry in one_shot_unavailable)
    invalid_query_count = sum("invalid boolean" in entry.get("error", "").lower() for entry in one_shot_unavailable)
    other_error_count = len(one_shot_unavailable) - no_proposal_count - invalid_query_count
    st.markdown('<div class="section-title">One shot Groq coverage</div>', unsafe_allow_html=True)
    st.caption(
        f"{len(one_shot_available)} topics have a valid proposal and {len(one_shot_unavailable)} topics are unavailable. "
        "Unavailable topics are excluded from the baseline averages."
    )
    availability_rows = []
    if no_proposal_count:
        availability_rows.append({"Unavailable reason": "No saved proposal", "Topics": no_proposal_count})
    if invalid_query_count:
        availability_rows.append({"Unavailable reason": "Invalid Boolean reply", "Topics": invalid_query_count})
    if other_error_count:
        availability_rows.append({"Unavailable reason": "Other sanitized error", "Topics": other_error_count})
    if availability_rows:
        st.dataframe(pd.DataFrame(availability_rows), hide_index=True, use_container_width=True)

    gold_entries = {str(entry["qid"]): entry for entry in data.get("goldilocks", [])}
    one_shot_by_qid = {str(entry["qid"]): entry for entry in one_shot_available}
    paired_qids = sorted(set(gold_entries) & set(one_shot_by_qid))
    if paired_qids:
        paired_gold = [gold_entries[qid] for qid in paired_qids]
        paired_one_shot = [one_shot_by_qid[qid] for qid in paired_qids]

        def paired_mean(entries, field):
            return sum(entries[index][field] for index in range(len(entries))) / len(entries)

        paired_rows = [
            {
                "Measure": "Topics compared",
                "Goldilocks": str(len(paired_qids)),
                "One shot Groq": str(len(paired_qids)),
            },
            {
                "Measure": "Target band rate",
                "Goldilocks": f"{paired_mean(paired_gold, 'in_band'):.1%}",
                "One shot Groq": f"{paired_mean(paired_one_shot, 'in_band'):.1%}",
            },
            {
                "Measure": "Average hits",
                "Goldilocks": f"{paired_mean(paired_gold, 'hits'):.1f}",
                "One shot Groq": f"{paired_mean(paired_one_shot, 'hits'):.1f}",
            },
            {
                "Measure": "Precision",
                "Goldilocks": f"{paired_mean(paired_gold, 'precision'):.4f}",
                "One shot Groq": f"{paired_mean(paired_one_shot, 'precision'):.4f}",
            },
            {
                "Measure": "Recall",
                "Goldilocks": f"{paired_mean(paired_gold, 'recall'):.4f}",
                "One shot Groq": f"{paired_mean(paired_one_shot, 'recall'):.4f}",
            },
            {
                "Measure": "P at 10",
                "Goldilocks": f"{paired_mean(paired_gold, 'p_at_10'):.4f}",
                "One shot Groq": f"{paired_mean(paired_one_shot, 'p_at_10'):.4f}",
            },
            {
                "Measure": "nDCG at 10",
                "Goldilocks": f"{paired_mean(paired_gold, 'ndcg_at_10'):.4f}",
                "One shot Groq": f"{paired_mean(paired_one_shot, 'ndcg_at_10'):.4f}",
            },
            {
                "Measure": "MRR at 10",
                "Goldilocks": f"{paired_mean(paired_gold, 'mrr_at_10'):.4f}",
                "One shot Groq": f"{paired_mean(paired_one_shot, 'mrr_at_10'):.4f}",
            },
        ]
        st.markdown('<div class="section-title">Paired comparison on the same topics</div>', unsafe_allow_html=True)
        st.caption("This comparison includes only topics with a valid one shot Groq proposal.")
        st.dataframe(pd.DataFrame(paired_rows), hide_index=True, use_container_width=True)

    available = [summary for summary in summaries if summary["Available"]]
    if available:
        st.markdown('<div class="section-title">Compare the systems</div>', unsafe_allow_html=True)
        grouped = []
        hit_values = []
        for summary in available:
            grouped.extend(
                [
                    {"System": summary["System"], "Measure": "In band rate", "Percent": summary["In band rate"] * 100},
                    {"System": summary["System"], "Measure": "P at 10", "Percent": summary["P at 10"] * 100},
                ]
            )
            hit_values.append(
                {
                    "System": summary["System"],
                    "Log hits": math.log10(max(summary["Average hits"], 0) + 1),
                    "Average hits": summary["Average hits"],
                }
            )

        grouped_chart = (
            alt.Chart(pd.DataFrame(grouped))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("System:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("Measure:N"),
                y=alt.Y("Percent:Q", title="Percent", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Measure:N", scale=alt.Scale(domain=["In band rate", "P at 10"], range=["#6C5CE7", "#14B8A6"]), legend=alt.Legend(orient="top")),
                tooltip=["System:N", "Measure:N", alt.Tooltip("Percent:Q", format=".1f")],
            )
            .properties(height=270)
        )
        hits_chart = (
            alt.Chart(pd.DataFrame(hit_values))
            .mark_bar(color="#3B82F6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("System:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Log hits:Q", title="Log10 of average hits plus one"),
                tooltip=["System:N", alt.Tooltip("Average hits:Q", format=",.1f")],
            )
            .properties(height=270)
        )
        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.markdown("#### In band rate and P at 10")
            st.altair_chart(grouped_chart, use_container_width=True)
        with chart_columns[1]:
            st.markdown("#### Result size on a log scale")
            st.altair_chart(hits_chart, use_container_width=True)

    st.markdown(
        '<div class="honesty-note">Hitting the target size measures size control, not relevance. '
        "Goldilocks recall is low in the saved evaluation, so a useful result count does not mean the relevant set is complete."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Full topic table</div>', unsafe_allow_html=True)
    table_rows = []
    for summary in summaries:
        if not summary["Available"]:
            table_rows.append(
                {
                    "System": summary["System"],
                    "Topics": "Not scored",
                    "In band rate": "Not scored",
                    "Average hits": "Not scored",
                    "Precision": "Not scored",
                    "Recall": "Not scored",
                    "P at 10": "Not scored",
                    "nDCG at 10": "Not scored",
                    "MRR at 10": "Not scored",
                }
            )
            continue
        table_rows.append(
            {
                "System": summary["System"],
                "Topics": summary["Topics"],
                "In band rate": f"{summary['In band rate']:.1%}",
                "Average hits": f"{summary['Average hits']:,.1f}",
                "Precision": f"{summary['Precision']:.4f}",
                "Recall": f"{summary['Recall']:.4f}",
                "P at 10": f"{summary['P at 10']:.4f}",
                "nDCG at 10": f"{summary['nDCG at 10']:.4f}",
                "MRR at 10": f"{summary['MRR at 10']:.4f}",
            }
        )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    with st.expander("View saved topic records"):
        st.json(data)


def render_pipeline():
    steps = [
        ("Corpus", "Collection documents and metadata"),
        ("Tokenizer and stemmer", "Token normalization"),
        ("Inverted index", "Dictionary and postings lists"),
        ("Dictionary with df", "Document frequency statistics"),
        ("Boolean engine", "AND, OR, and NOT set operations"),
        ("Controller", "Query processing and reformulation"),
        ("TF IDF ranker", "Vector space ranking and cosine"),
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
    search_tab, evaluation_tab, guide_tab = st.tabs(["Search", "Evaluation", "How it works"])
    with search_tab:
        render_search(root, index, config)
    with evaluation_tab:
        render_evaluation(root)
    with guide_tab:
        render_how_it_works(config)
