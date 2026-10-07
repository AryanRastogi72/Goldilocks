import json
from pathlib import Path

import pandas as pd
import streamlit as st

from engine.controller import run_controller
from engine.gemini_client import using_api_key
from engine.search import SearchIndex
from pipeline.tokenizer import tokenize


ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading the search index")
def load_index():
    return SearchIndex(str(ROOT / "index_out"), max_cached_shards=1)


def secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def show_search(index):
    st.subheader("Search the COVID research collection")
    query = st.text_input("Describe what you want to find", value="COVID vaccine pregnancy")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    with st.sidebar:
        st.header("Search settings")
        mode = st.radio("Query proposal", ["Offline keywords", "Gemini"]) 
        key_input = ""
        if mode == "Gemini":
            key_input = st.text_input("Gemini API key", type="password", key="gemini_api_key")
        low = st.number_input("Minimum results", min_value=1, value=int(config.get("target_band_low", 20)))
        high = st.number_input("Maximum results", min_value=1, value=int(config.get("target_band_high", 200)))
        iterations = st.number_input("Iteration limit", min_value=1, max_value=20,
                                     value=int(config.get("max_iterations", 8)))
        top_k = st.number_input("Results to rank", min_value=1, max_value=100,
                                value=int(config.get("top_k", 10)))

    if low > high:
        st.error("The minimum must be less than or equal to the maximum.")
        return
    if not st.button("Search", type="primary") or not query.strip():
        return

    config.update({"target_band_low": int(low), "target_band_high": int(high),
                   "max_iterations": int(iterations), "top_k": int(top_k)})
    chosen_key = key_input.strip() or secret_key()
    use_gemini = mode == "Gemini" and bool(chosen_key)
    if mode == "Gemini" and not chosen_key:
        st.warning("No key was entered. Searching with offline keywords.")

    with st.spinner("Running the Boolean controller and ranking results"):
        if use_gemini:
            with using_api_key(chosen_key):
                ranked, trace, final_query = run_controller(query, index, config, use_llm=True)
            if trace.llm_errors:
                st.warning("Gemini was unavailable. The search was repeated in offline mode. " + trace.llm_errors[0])
                ranked, trace, final_query = run_controller(query, index, config, use_llm=False)
        else:
            ranked, trace, final_query = run_controller(query, index, config, use_llm=False)

    st.caption(f"Collection size: {index.num_docs:,} documents. Final query: {final_query}")
    st.markdown("#### Query version timeline")
    rows = trace.iterations
    if rows:
        st.dataframe(pd.DataFrame([{
            "Version": row["iteration"], "Boolean query": row["query"],
            "Decision": row["decision"], "Estimated bound": row["estimated_hits"],
            "Actual hits": row["actual_hits"], "Reason": row["reason"]
        } for row in rows]), hide_index=True, width="stretch")
        if len(rows) > 1:
            st.bar_chart(pd.DataFrame([{
                "Version": row["iteration"], "Estimated bound": row["estimated_hits"],
                "Actual hits": row["actual_hits"]
            } for row in rows]).set_index("Version"))

        latest = rows[-1]
        st.markdown("#### Document frequency table")
        df_rows = [{"Term": term, "Document frequency": int(df)} for term, df in latest["df_table"].items()]
        if df_rows:
            st.dataframe(pd.DataFrame(df_rows), hide_index=True, width="stretch",
                         column_config={"Document frequency": st.column_config.ProgressColumn(
                             "Document frequency", min_value=0,
                             max_value=max(1, max(df["Document frequency"] for df in df_rows)))})
        st.write(f"Estimated bound: {latest['estimated_hits']:,}, actual hits: {latest['actual_hits']:,}, target: {low} to {high}")

    st.markdown("#### Postings preview")
    query_terms = [term for term in tokenize(final_query) if term not in {"and", "or", "not"}]
    posting_rows = []
    for term in dict.fromkeys(query_terms):
        for doc_id, tf in index.get_postings(term)[:5]:
            posting_rows.append({"Term": term, "Document ID": index.get_ext_id(doc_id), "Term frequency": tf})
    if posting_rows:
        st.dataframe(pd.DataFrame(posting_rows), hide_index=True, width="stretch")
    else:
        st.info("No postings were found for the final query terms.")

    st.markdown("#### Ranked results")
    if ranked:
        result_rows = [{"Rank": rank, "Score": round(float(score), 6),
                        "Document ID": index.get_ext_id(doc_id),
                        "Title": index.get_doc_title(doc_id)}
                       for rank, (score, doc_id) in enumerate(ranked, start=1)]
        st.dataframe(pd.DataFrame(result_rows), hide_index=True, width="stretch")
    else:
        st.info("The Boolean query returned no documents.")


def show_evaluation():
    st.subheader("Saved evaluation")
    path = ROOT / "evaluation" / "results.json"
    if not path.exists():
        st.info("No saved evaluation results are available yet.")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    methods = [
        ("Goldilocks", "goldilocks"),
        ("One shot Gemini", "one_shot_llm_baseline"),
        ("AND keywords", "and_baseline"),
        ("OR keywords", "or_baseline"),
    ]
    summary = []
    chart_rows = []
    for label, key in methods:
        entries = [entry for entry in data.get(key, []) if entry.get("available", True)]
        if not entries:
            summary.append({"Method": label, "Evaluated topics": 0, "Result": "Not available"})
            continue
        count = len(entries)
        in_band = sum(bool(entry["in_band"]) for entry in entries)
        row = {"Method": label, "Evaluated topics": count,
               "In target band": f"{in_band} of {count}",
               "Average hits": round(sum(entry["hits"] for entry in entries) / count, 1),
               "Average precision": round(sum(entry["precision"] for entry in entries) / count, 4),
               "Average recall": round(sum(entry["recall"] for entry in entries) / count, 4),
               "Average P at 10": round(sum(entry["p_at_10"] for entry in entries) / count, 4)}
        if key == "goldilocks":
            row["Average controller steps"] = round(sum(entry.get("iterations", 0) for entry in entries) / count, 2)
            row["Average Gemini calls"] = round(sum(entry.get("llm_calls", 0) for entry in entries) / count, 2)
        summary.append(row)
        chart_rows.append({"Method": label, "Average hits": row["Average hits"],
                           "Average P at 10": row["Average P at 10"]})
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    if chart_rows:
        chart_frame = pd.DataFrame(chart_rows).set_index("Method")
        st.markdown("#### Average hits")
        st.bar_chart(chart_frame[["Average hits"]])
        st.markdown("#### Average precision at 10")
        st.bar_chart(chart_frame[["Average P at 10"]])
    with st.expander("View saved topic level records"):
        st.json(data)


def show_concepts():
    st.subheader("How the search connects to information retrieval")
    lessons = pd.DataFrame([
        {"Project component": "Tokenizer and stemmer", "Lecture concept": "Tokenization and normalization", "What it does": "Maps words to normalized index terms."},
        {"Project component": "Inverted index", "Lecture concept": "Dictionary and postings", "What it does": "Finds documents and term counts without scanning the collection."},
        {"Project component": "K gram index", "Lecture concept": "Wildcard vocabulary lookup", "What it does": "Finds matching terms from short character sequences."},
        {"Project component": "Boolean engine", "Lecture concept": "AND, OR, NOT and set operations", "What it does": "Combines postings into a candidate document set."},
        {"Project component": "DF ordered AND", "Lecture concept": "Query processing optimization", "What it does": "Intersects smaller postings lists first."},
        {"Project component": "Controller", "Lecture concept": "Collection statistics and query reformulation", "What it does": "Uses df bounds and actual hit counts to tighten or relax."},
        {"Project component": "Ranker", "Lecture concept": "TF IDF and cosine similarity", "What it does": "Orders the Boolean matches by term based similarity."},
        {"Project component": "Evaluation", "Lecture concept": "Precision, recall and P at 10", "What it does": "Compares results with relevance judgments."},
    ])
    st.dataframe(lessons, hide_index=True, width="stretch")


st.set_page_config(page_title="Goldilocks Search", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background: #edf3ef;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #dce7e1; padding: 1rem; border-radius: 12px;}
button[kind="primary"] {border-radius: 8px; font-weight: 650;}
[data-baseweb="tab"] {font-weight: 600;}
*:focus-visible {outline: 3px solid #176B63; outline-offset: 2px;}
::selection {background: #c6e8df; color: #18302C;}
</style>
""", unsafe_allow_html=True)
st.title("Goldilocks")
st.caption("Boolean search guided by collection statistics")
tabs = st.tabs(["Search", "Evaluation", "How it works"])
try:
    with tabs[0]:
        show_search(load_index())
    with tabs[1]:
        show_evaluation()
    with tabs[2]:
        show_concepts()
except FileNotFoundError as error:
    st.error(f"Required project data is missing: {Path(error.filename or '').name}. Build the index before starting the app.")
