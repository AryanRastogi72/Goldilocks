import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from engine.controller import run_controller
from engine.llm_client import using_api_key
from engine.search import SearchIndex
from pipeline.tokenizer import tokenize


ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Goldilocks Search",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the collection index")
def load_index():
    return SearchIndex(str(ROOT / "index_out"), max_cached_shards=1)


def secret_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""


def safe(value):
    return html.escape(str(value), quote=True)


def add_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #18332c;
            --ink-soft: #466057;
            --paper: #f4f2eb;
            --surface: #fffefa;
            --surface-muted: #e8ece5;
            --line: #d8ded5;
            --teal: #176b5d;
            --teal-dark: #105548;
            --teal-pale: #d9eee5;
            --coral: #bd654e;
            --coral-pale: #f7e6dd;
            --gold: #896d2d;
            --gold-pale: #f3ecd5;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--paper);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1380px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: #e9ede5;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin: 0 0 1.5rem 0;
        }

        .brand-mark {
            display: grid;
            place-items: center;
            width: 2.65rem;
            height: 2.65rem;
            border-radius: 0.85rem;
            background: var(--ink);
            color: #f8f5ec;
            font-family: Georgia, serif;
            font-size: 1.55rem;
            font-weight: 700;
        }

        .brand-name {
            color: var(--ink);
            font-size: 1.2rem;
            font-weight: 720;
            letter-spacing: -0.035em;
            line-height: 1.15;
        }

        .brand-note {
            color: var(--ink-soft);
            font-size: 0.78rem;
            margin-top: 0.18rem;
        }

        .page-title {
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.25rem);
            font-weight: 700;
            letter-spacing: -0.055em;
            line-height: 1.04;
            margin: 0;
        }

        .page-intro {
            max-width: 53rem;
            color: var(--ink-soft);
            font-size: 1.05rem;
            line-height: 1.65;
            margin: 0.8rem 0 1.2rem 0;
        }

        .collection-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
            color: var(--ink-soft);
            font-size: 0.82rem;
            margin: 0.4rem 0 1.6rem 0;
        }

        .collection-chip {
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgb(255 254 250 / 72%);
            color: var(--ink);
            padding: 0.35rem 0.7rem;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 1.4rem;
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            height: 3.2rem;
            padding: 0 0.2rem;
            color: var(--ink-soft);
            font-size: 0.94rem;
            font-weight: 620;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            color: var(--teal-dark);
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: var(--teal);
            height: 3px;
            border-radius: 3px 3px 0 0;
        }

        h2, h3, h4 {
            color: var(--ink);
            letter-spacing: -0.025em;
        }

        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stRadio"] label {
            color: var(--ink);
            font-weight: 620;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            min-height: 3rem;
            border-color: #c8d2c8;
            border-radius: 0.7rem;
            background: var(--surface);
            color: var(--ink);
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus {
            border-color: var(--teal);
            box-shadow: 0 0 0 2px rgb(23 107 93 / 15%);
        }

        button[kind="primary"] {
            min-height: 3rem;
            border: 1px solid var(--teal);
            border-radius: 0.7rem;
            background: var(--teal);
            color: white;
            font-weight: 680;
            transition: background-color 140ms ease, transform 140ms ease;
        }

        button[kind="primary"]:hover {
            border-color: var(--teal-dark);
            background: var(--teal-dark);
            transform: translateY(-1px);
        }

        button[kind="primary"]:active {
            transform: translateY(0);
        }

        [data-testid="stMetric"] {
            min-height: 6.2rem;
            padding: 1rem 1.1rem;
            border: 1px solid var(--line);
            border-radius: 0.85rem;
            background: var(--surface);
        }

        [data-testid="stMetricLabel"] {
            color: var(--ink-soft);
            font-size: 0.8rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-variant-numeric: tabular-nums;
            font-size: 1.55rem;
            font-weight: 700;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 0.7rem;
            overflow: hidden;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 0.75rem;
            background: rgb(255 254 250 / 70%);
        }

        [data-testid="stAlert"] {
            border-radius: 0.7rem;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--ink-soft);
        }

        .section-heading {
            margin: 1.8rem 0 0.25rem 0;
            color: var(--ink);
            font-size: 1.28rem;
            font-weight: 690;
            letter-spacing: -0.035em;
        }

        .section-note {
            margin: 0 0 0.9rem 0;
            color: var(--ink-soft);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .trace-list {
            display: grid;
            gap: 0.55rem;
            margin: 0.5rem 0 1rem 0;
        }

        .trace-row {
            display: grid;
            grid-template-columns: 2.25rem minmax(12rem, 1fr) auto;
            gap: 0.8rem;
            align-items: start;
            padding: 0.9rem 1rem;
            border: 1px solid var(--line);
            border-radius: 0.75rem;
            background: var(--surface);
        }

        .trace-step {
            display: grid;
            place-items: center;
            width: 2rem;
            height: 2rem;
            border-radius: 0.6rem;
            background: var(--surface-muted);
            color: var(--ink);
            font-size: 0.78rem;
            font-variant-numeric: tabular-nums;
            font-weight: 700;
        }

        .trace-query {
            overflow-wrap: anywhere;
            color: var(--ink);
            font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
            font-size: 0.83rem;
            line-height: 1.55;
        }

        .trace-reason {
            margin-top: 0.35rem;
            color: var(--ink-soft);
            font-size: 0.79rem;
            line-height: 1.45;
        }

        .trace-counts {
            margin-top: 0.4rem;
            color: var(--ink-soft);
            font-size: 0.75rem;
            font-variant-numeric: tabular-nums;
        }

        .decision {
            white-space: nowrap;
            border-radius: 999px;
            padding: 0.3rem 0.55rem;
            color: var(--ink);
            background: var(--surface-muted);
            font-size: 0.68rem;
            font-weight: 720;
            letter-spacing: 0.025em;
        }

        .decision-accept {
            color: #18513f;
            background: var(--teal-pale);
        }

        .decision-tighten {
            color: #824331;
            background: var(--coral-pale);
        }

        .decision-relax {
            color: #705a20;
            background: var(--gold-pale);
        }

        .concept-list {
            display: grid;
            margin-top: 1rem;
            border-top: 1px solid var(--line);
        }

        .concept-row {
            display: grid;
            grid-template-columns: 2.5rem minmax(11rem, 0.9fr) minmax(11rem, 1fr) minmax(14rem, 1.4fr);
            gap: 1rem;
            align-items: start;
            padding: 1.1rem 0.4rem;
            border-bottom: 1px solid var(--line);
        }

        .concept-step {
            color: var(--coral);
            font-family: Georgia, serif;
            font-size: 1.1rem;
            font-variant-numeric: tabular-nums;
        }

        .concept-component {
            color: var(--ink);
            font-weight: 680;
        }

        .concept-lecture {
            color: var(--teal-dark);
            font-size: 0.88rem;
            font-weight: 620;
        }

        .concept-purpose {
            color: var(--ink-soft);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .side-note {
            padding: 0.75rem 0.8rem;
            border-radius: 0.7rem;
            background: rgb(255 254 250 / 76%);
            color: var(--ink-soft);
            font-size: 0.82rem;
            line-height: 1.5;
        }

        *:focus-visible {
            outline: 3px solid var(--teal);
            outline-offset: 3px;
        }

        ::selection {
            background: #b8dfd1;
            color: var(--ink);
        }

        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] {
                padding: 1.1rem 1rem 2.5rem 1rem;
            }

            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.7rem;
            }

            [data-testid="column"] {
                flex: 1 1 12rem !important;
                min-width: min(100%, 12rem) !important;
            }

            .page-title {
                font-size: 2.2rem;
            }

            .trace-row {
                grid-template-columns: 2rem minmax(0, 1fr);
                gap: 0.55rem;
            }

            .decision {
                grid-column: 2;
                justify-self: start;
            }

            .concept-row {
                grid-template-columns: 2rem minmax(0, 1fr);
                gap: 0.35rem 0.75rem;
            }

            .concept-lecture,
            .concept-purpose {
                grid-column: 2;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(index, config):
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">G</div>
            <div>
                <div class="brand-name">Goldilocks</div>
                <div class="brand-note">Information retrieval project</div>
            </div>
        </div>
        <h1 class="page-title">Find the useful middle.</h1>
        <p class="page-intro">
            Turn a research question into a Boolean search, then follow the
            collection statistics as the controller steers toward a useful
            number of papers.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="collection-strip">
            <span class="collection-chip">{index.num_docs:,} indexed documents</span>
            <span class="collection-chip">Target range {config["target_band_low"]} to {config["target_band_high"]} results</span>
            <span class="collection-chip">TREC COVID collection</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_style(decision):
    if decision == "ACCEPT":
        return "decision-accept"
    if "TIGHTEN" in decision:
        return "decision-tighten"
    if "RELAX" in decision:
        return "decision-relax"
    return ""


def render_trace(rows):
    items = []
    for row in rows:
        decision = safe(row.get("decision", ""))
        style = decision_style(str(row.get("decision", "")))
        items.append(
            f'<div class="trace-row"><div class="trace-step">{int(row.get("iteration", 0)):02d}</div>'
            f'<div><div class="trace-query">{safe(row.get("query", ""))}</div>'
            f'<div class="trace-reason">{safe(row.get("reason", ""))}</div>'
            f'<div class="trace-counts">Estimate {int(row.get("estimated_hits", 0)):,} '
            f'&nbsp; · &nbsp; Actual {int(row.get("actual_hits", 0)):,}</div></div>'
            f'<div><span class="decision {style}">{decision}</span></div></div>'
        )
    st.markdown(
        f'<div class="trace-list">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )


def run_search(query, index, config, mode):
    chosen_key = secret_key()
    use_groq = mode == "Groq suggestions" and bool(chosen_key or os.environ.get("GROQ_API_KEY"))
    if mode == "Groq suggestions" and not use_groq:
        st.warning("Groq is not configured. The search will use offline keywords.")

    with st.spinner("Reading the index and ranking matching papers"):
        if use_groq:
            with using_api_key(chosen_key):
                ranked, trace, final_query = run_controller(query, index, config, use_llm=True)
            if trace.llm_errors:
                st.warning("Groq could not return a proposal. The search was repeated with offline keywords.")
                ranked, trace, final_query = run_controller(query, index, config, use_llm=False)
                mode_used = "Offline keywords"
            else:
                mode_used = "Groq suggestions"
        else:
            ranked, trace, final_query = run_controller(query, index, config, use_llm=False)
            mode_used = "Offline keywords"

    st.session_state["goldilocks_result"] = {
        "query": query,
        "ranked": ranked,
        "trace": trace,
        "final_query": final_query,
        "mode_used": mode_used,
        "settings": config.copy(),
    }


def show_search(index, base_config):
    with st.sidebar:
        st.markdown("## Search settings")
        st.caption("Groq proposes terms. The index statistics guide every controller decision.")
        mode = st.radio(
            "Query proposal",
            ["Offline keywords", "Groq suggestions"],
            label_visibility="visible",
        )
        low = st.number_input(
            "Minimum results",
            min_value=1,
            value=int(base_config.get("target_band_low", 20)),
        )
        high = st.number_input(
            "Maximum results",
            min_value=1,
            value=int(base_config.get("target_band_high", 200)),
        )
        iterations = st.number_input(
            "Iteration limit",
            min_value=1,
            max_value=20,
            value=int(base_config.get("max_iterations", 8)),
        )
        top_k = st.number_input(
            "Ranked papers",
            min_value=1,
            max_value=100,
            value=int(base_config.get("top_k", 10)),
        )
        st.markdown(
            '<div class="side-note">The Groq key is held by the app. Visitors do not need to enter a key.</div>',
            unsafe_allow_html=True,
        )

    with st.form("goldilocks_search_form"):
        query_col, button_col = st.columns([0.78, 0.22], vertical_alignment="bottom")
        with query_col:
            query = st.text_input(
                "Research question",
                value="COVID vaccine pregnancy",
                placeholder="For example, long term symptoms after COVID",
            )
        with button_col:
            submitted = st.form_submit_button(
                "Search collection",
                type="primary",
                use_container_width=True,
            )

    if low > high:
        st.error("The minimum result count must not exceed the maximum.")
        return

    if submitted:
        if not query.strip():
            st.warning("Enter a research question to begin.")
        else:
            config = base_config.copy()
            config.update(
                {
                    "target_band_low": int(low),
                    "target_band_high": int(high),
                    "max_iterations": int(iterations),
                    "top_k": int(top_k),
                }
            )
            run_search(query.strip(), index, config, mode)

    result = st.session_state.get("goldilocks_result")
    if result is None:
        st.markdown(
            '<p class="section-note">Submit a question to see its query path, document frequencies, postings, and ranked papers.</p>',
            unsafe_allow_html=True,
        )
        return

    trace = result["trace"]
    rows = trace.iterations
    final_hits = rows[-1]["actual_hits"] if rows else 0
    target = result["settings"]
    st.caption(
        f'Last search: {result["query"]} · Proposal mode: {result["mode_used"]} · '
        f'Final Boolean query: {result["final_query"]}'
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Matching papers", f"{final_hits:,}")
    metric_cols[1].metric(
        "Target range",
        f'{target["target_band_low"]} to {target["target_band_high"]}',
    )
    metric_cols[2].metric("Controller steps", len(rows))
    metric_cols[3].metric("Groq calls", trace.llm_calls)

    st.markdown('<div class="section-heading">Controller trace</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">Each line shows the Boolean query, the controller action, its estimate, and the result count from the index.</p>',
        unsafe_allow_html=True,
    )
    if rows:
        render_trace(rows)
        chart_rows = [
            {
                "Version": row["iteration"],
                "Estimated bound": row["estimated_hits"],
                "Actual hits": row["actual_hits"],
                "Minimum target": target["target_band_low"],
                "Maximum target": target["target_band_high"],
            }
            for row in rows
        ]
        with st.expander("View result size across controller steps"):
            chart = pd.DataFrame(chart_rows).set_index("Version")
            st.line_chart(chart)
    else:
        st.info("The controller did not produce a query trace.")

    df_rows = [
        {"Term": term, "Document frequency": int(df)}
        for term, df in (rows[-1].get("df_table", {}).items() if rows else [])
    ]
    query_terms = [
        term
        for term in tokenize(result["final_query"])
        if term not in {"and", "or", "not"}
    ]
    posting_rows = []
    for term in dict.fromkeys(query_terms):
        for doc_id, tf in index.get_postings(term)[:5]:
            posting_rows.append(
                {
                    "Term": term,
                    "Document ID": index.get_ext_id(doc_id),
                    "Term frequency": tf,
                }
            )

    st.markdown('<div class="section-heading">Index evidence</div>', unsafe_allow_html=True)
    evidence_cols = st.columns(2, gap="large")
    with evidence_cols[0]:
        st.markdown("#### Document frequency")
        st.caption("How many collection documents contain each term.")
        if df_rows:
            st.dataframe(
                pd.DataFrame(df_rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Document frequency": st.column_config.ProgressColumn(
                        "Document frequency",
                        min_value=0,
                        max_value=max(1, max(row["Document frequency"] for row in df_rows)),
                    )
                },
            )
        else:
            st.info("No term statistics are available for this trace.")
    with evidence_cols[1]:
        st.markdown("#### Postings sample")
        st.caption("A few document IDs and term counts from the index.")
        if posting_rows:
            st.dataframe(
                pd.DataFrame(posting_rows),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No postings were found for the final query terms.")

    st.markdown('<div class="section-heading">Ranked papers</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">The Boolean engine finds the candidates. TF IDF cosine similarity orders them.</p>',
        unsafe_allow_html=True,
    )
    if result["ranked"]:
        result_rows = [
            {
                "Rank": rank,
                "Score": round(float(score), 6),
                "Document ID": index.get_ext_id(doc_id),
                "Title": index.get_doc_title(doc_id),
            }
            for rank, (score, doc_id) in enumerate(result["ranked"], start=1)
        ]
        st.dataframe(
            pd.DataFrame(result_rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Score": st.column_config.NumberColumn("TF IDF cosine score", format="%.4f"),
                "Document ID": st.column_config.TextColumn("Document ID", width="small"),
                "Title": st.column_config.TextColumn("Paper title", width="large"),
            },
        )
    else:
        st.info("The Boolean query returned no documents.")


def show_evaluation():
    st.markdown('<div class="section-heading">Saved evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">Measured results from the fifty topic benchmark and its relevance judgments.</p>',
        unsafe_allow_html=True,
    )
    path = ROOT / "evaluation" / "results.json"
    if not path.exists():
        st.info("No saved evaluation results are available yet.")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    methods = [
        ("Goldilocks", "goldilocks"),
        ("One shot Groq", "one_shot_llm_baseline"),
        ("AND keywords", "and_baseline"),
        ("OR keywords", "or_baseline"),
    ]
    summary = []
    chart_rows = []
    for label, key in methods:
        entries = [entry for entry in data.get(key, []) if entry.get("available", True)]
        if not entries:
            summary.append(
                {"Method": label, "Evaluated topics": 0, "Result": "Not available"}
            )
            continue
        count = len(entries)
        in_band = sum(bool(entry["in_band"]) for entry in entries)
        row = {
            "Method": label,
            "Evaluated topics": count,
            "In target band": f"{in_band} of {count}",
            "Average hits": round(sum(entry["hits"] for entry in entries) / count, 1),
            "Average precision": round(
                sum(entry["precision"] for entry in entries) / count, 4
            ),
            "Average recall": round(
                sum(entry["recall"] for entry in entries) / count, 4
            ),
            "Average P at 10": round(
                sum(entry["p_at_10"] for entry in entries) / count, 4
            ),
        }
        if key == "goldilocks":
            row["Average controller steps"] = round(
                sum(entry.get("iterations", 0) for entry in entries) / count, 2
            )
            row["Average model calls"] = round(
                sum(entry.get("llm_calls", 0) for entry in entries) / count, 2
            )
        summary.append(row)
        chart_rows.append(
            {
                "Method": label,
                "Average hits": row["Average hits"],
                "Average precision": row["Average precision"],
                "Average recall": row["Average recall"],
                "Average P at 10": row["Average P at 10"],
            }
        )

    goldilocks = next(
        (row for row in summary if row.get("Method") == "Goldilocks"),
        None,
    )
    if goldilocks and goldilocks.get("Evaluated topics", 0):
        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Topics in target range",
            goldilocks["In target band"],
        )
        metric_cols[1].metric("Average precision", goldilocks["Average precision"])
        metric_cols[2].metric("Average recall", goldilocks["Average recall"])
        metric_cols[3].metric("Average P at 10", goldilocks["Average P at 10"])

    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
    if chart_rows:
        chart_col, note_col = st.columns([1.15, 0.85], gap="large")
        with chart_col:
            chart_frame = pd.DataFrame(chart_rows).set_index("Method")
            st.markdown("#### Average returned papers")
            st.bar_chart(chart_frame[["Average hits"]])
            st.markdown("#### Relevance measures")
            st.bar_chart(
                chart_frame[
                    ["Average precision", "Average recall", "Average P at 10"]
                ]
            )
        with note_col:
            st.markdown("#### Reading the scores")
            st.write(
                "Precision measures how many returned papers were relevant. "
                "Recall measures how much of the judged relevant set was found. "
                "P at 10 measures relevance in the first ten ranked papers."
            )

    st.caption(
        "The saved topic evaluation predates the Groq provider change. "
        "The one shot Groq baseline is not available in these saved results."
    )
    with st.expander("View the saved topic records"):
        st.json(data)


def show_concepts():
    st.markdown('<div class="section-heading">From question to ranked papers</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">Each stage connects a project component to a familiar information retrieval idea.</p>',
        unsafe_allow_html=True,
    )
    lessons = [
        ("Tokenizer and stemmer", "Tokenization and normalization", "Maps words to normalized index terms."),
        ("Inverted index", "Dictionary and postings", "Finds documents and term counts without scanning the collection."),
        ("K gram index", "Wildcard vocabulary lookup", "Finds matching terms from short character sequences."),
        ("Boolean engine", "AND, OR, NOT and set operations", "Combines postings into a candidate document set."),
        ("DF ordered AND", "Query processing optimization", "Intersects smaller postings lists first."),
        ("Controller", "Collection statistics and reformulation", "Uses df bounds and actual hit counts to tighten or relax."),
        ("Ranker", "TF IDF and cosine similarity", "Orders the Boolean matches by term similarity."),
        ("Evaluation", "Precision, recall and P at 10", "Compares results with relevance judgments."),
    ]
    items = []
    for step, (component, lecture, purpose) in enumerate(lessons, start=1):
        items.append(
            f'<div class="concept-row"><div class="concept-step">{step:02d}</div>'
            f'<div class="concept-component">{safe(component)}</div>'
            f'<div class="concept-lecture">{safe(lecture)}</div>'
            f'<div class="concept-purpose">{safe(purpose)}</div></div>'
        )
    st.markdown(
        f'<div class="concept-list">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-note" style="margin-top:1.2rem">Groq can suggest query terms. It does not choose when a query is accepted, tightened, or relaxed.</p>',
        unsafe_allow_html=True,
    )


def main():
    add_styles()
    index = load_index()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    render_header(index, config)
    search_tab, evaluation_tab, concepts_tab = st.tabs(
        ["Search", "Evaluation", "How it works"]
    )
    with search_tab:
        show_search(index, config)
    with evaluation_tab:
        show_evaluation()
    with concepts_tab:
        show_concepts()


try:
    main()
except FileNotFoundError as error:
    st.error(
        f"Required project data is missing: {Path(error.filename or '').name}. "
        "Build the index before starting the app."
    )
