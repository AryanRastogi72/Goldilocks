# Goldilocks

Search results that are just the right size.

Goldilocks is an Information Retrieval course project for scientific search. It turns a research question into a Boolean query, checks the index statistics, adjusts the query when the result set is too large or too small, then ranks the matching documents with tf idf cosine similarity.

## Track choice

Our main track is Track 2, Conversational and agentic search. The app keeps recent questions and accepted queries for the current browser session. A follow up can use that context when Groq is enabled. Offline mode keeps the earlier query and adds terms from the follow up, then the same controller adjusts the result size. Session context is temporary and does not persist after the browser session ends.

## Try the hosted app

[Open Goldilocks](https://goldilocks-hackathon.streamlit.app/)

The project owner reports that the deployed app and public repository are working. The owner configures the Groq secret. Visitors can also use Offline mode without a key.

## Novelty

The project contribution is an inspectable query size controller for a large scientific collection. It uses postings document frequencies to estimate the likely result size, checks the exact Boolean hit count, and records each accept, tighten or relax decision. The app also carries recent search context into follow up questions during the current session. These are system integration and teaching contributions built from established information retrieval methods. They are not a new retrieval algorithm, and the evaluation does not show that Goldilocks is more relevant overall.

## How search works

1. The tokenizer normalizes the question and stems terms.
2. Groq may propose a Boolean query using the current question and recent session context. It proposes terms only, it does not decide whether to tighten or relax.
3. The controller reads document frequency, estimates possible result counts, runs Boolean retrieval and checks the actual hit count.
4. The controller accepts a result count in the configured target band, tightens a query above the band, or relaxes a query below the band.
5. The ranker orders the matching papers by tf idf cosine score and returns the requested top K.

The default target band is 20 through 200 results. The app also works in Offline mode with keyword queries. Successful Groq proposals are cached locally. A cache miss without a key falls back to Offline mode.

## Project map

1. `app.py`, Streamlit entry point.
2. `ui.py` and `ui_styles.py`, interface and visual styling.
3. `pipeline`, corpus loading, tokenization, stemming and index building.
4. `engine`, Boolean parsing, postings search, the controller, Groq requests and tf idf ranking.
5. `index_out`, the prebuilt index for 171,332 documents.
6. `evaluation`, saved topic results, charts, summary and metric validator.
7. `config.json`, model, target band and search settings.

Private notes, secrets, raw corpus data, local caches and submission drafts are excluded by `.gitignore`.

## Run on Windows

Open PowerShell in the project folder, then run:

```powershell
python -m venv .venv
& '.venv\Scripts\Activate.ps1'
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = '1'
streamlit run app.py
```

If script activation is blocked in that PowerShell window, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& '.venv\Scripts\Activate.ps1'
```

For the command line reference search, run:

```powershell
$env:PYTHONUTF8 = '1'
& '.venv\Scripts\python.exe' -m engine.main 'covid vaccine pregnant'
```

For an evaluation using saved Groq replies and Offline fallback, remove the key from the current session first:

```powershell
Remove-Item Env:GROQ_API_KEY -ErrorAction SilentlyContinue
& '.venv\Scripts\python.exe' -m evaluation.run_eval
```

To build the index again, place the TREC COVID data in the expected local data folder, then run:

```powershell
& '.venv\Scripts\python.exe' -m pipeline.build_index
```

The index is already included, so most users do not need to download or rebuild it. The index occupies 208,118,057 bytes across 38 files. The largest individual file is 26,117,615 bytes. The saved full retrieval identifiers for evaluation are compressed in `evaluation/retrieved_ids.json.gz`.

The full evaluation and validator need `data/cache/queries.json` and `data/cache/qrels.json`. These local files are excluded from Git. To fetch the source topics and judgments, run the index pipeline command above. It also rebuilds the index. The prebuilt index is already present, so a first time app visitor does not need this step to search.

For Streamlit Community Cloud, select Python 3.12 in Advanced settings and put `GROQ_API_KEY` in the app Secrets field. The repository contains `runtime.txt` as a version note. Current Community Cloud setup selects Python in its deployment settings, so the file does not replace that selection. The local requirements install passed in a clean Python 3.13.5 environment. The project owner reports that the deployed app works.

## Verified evaluation

The values below are from the regenerated `evaluation/results.json` and `evaluation/summary.tsv`. P at 10 always divides the number of relevant items in the ranked first ten by ten, including when fewer than ten results were returned. nDCG at 10 uses graded relevance and discounts lower ranked results. MRR at 10 measures the position of the first relevant result.

1. Goldilocks, 50 topics, 43 in the target band, mean hits 73.5, precision 0.3286, recall 0.0535, P at 10 0.4860, nDCG at 10 0.4549, MRR at 10 0.6593.
2. One shot Groq Boolean baseline, 36 topics with valid proposals, 12 in the target band, mean hits 1809.0, precision 0.4863, recall 0.1491, P at 10 0.5333, nDCG at 10 0.5348, MRR at 10 0.7650.
3. AND keyword baseline, 50 topics, 5 in the target band, mean hits 8.9, precision 0.1481, recall 0.0089, P at 10 0.1280, nDCG at 10 0.1130, MRR at 10 0.2467.
4. OR keyword baseline, 50 topics, none in the target band, mean hits 135330.3, precision 0.0036, recall 0.9728, P at 10 0.2880, nDCG at 10 0.2542, MRR at 10 0.4525.

The Groq baseline is unavailable on 14 topics. In this key free replay, 11 had no cached proposal and 3 had invalid Boolean replies. Its fair direct comparison uses the same 36 topics for both systems. On those topics, Goldilocks has precision 0.3816, recall 0.0677, P at 10 0.5583, nDCG at 10 0.5198, MRR at 10 0.6907 and target band rate 80.56 percent. The one shot Groq baseline has precision 0.4863, recall 0.1491, P at 10 0.5333, nDCG at 10 0.5348, MRR at 10 0.7650 and target band rate 33.33 percent.

These results support the claim that Goldilocks controls result size more reliably. They do not show better overall relevance. On the paired topics, the one shot Groq baseline has higher precision, recall, nDCG at 10 and MRR at 10. Goldilocks has slightly higher P at 10 and a much higher paired target band rate. The OR baseline has much higher recall and very low precision. These systems make different tradeoffs.

Run the independent saved metric check with:

```powershell
& '.venv\Scripts\python.exe' -m evaluation.validate_saved_results
```

## Information Retrieval choices

1. The inverted index maps each term to a postings list of documents and term counts.
2. Document frequency supports hit estimates and orders AND processing from the shortest postings list.
3. The Boolean engine supports AND, OR, NOT and wildcard terms through a trigram k gram index.
4. The ranker uses logarithmic term frequency, inverse document frequency, vector length normalization, cosine similarity and heap based top K selection.
5. The evaluation uses the provided TREC COVID topics and relevance judgments, measuring precision, recall, P at 10, graded nDCG at 10 and MRR at 10.

## Libraries

1. NLTK supplies Porter stemming.
2. NumPy is installed as numeric support for the Python package stack. The project modules do not call it directly.
3. Requests sends Groq API requests.
4. Streamlit builds the interface.
5. Pandas formats evaluation and trace tables.
6. Altair renders evaluation charts.
7. Matplotlib writes saved evaluation charts.
8. Tabulate writes the text summary table.

## Data and related work

The corpus and judgments come from TREC COVID, a research collection created to evaluate search over scientific COVID 19 literature. The local index contains 171,332 documents. BEIR distributes TREC COVID as one of its information retrieval benchmark datasets.

Roberts and colleagues, “Searching for Scientific Evidence in a Pandemic: An Overview of TREC COVID,” Journal of Biomedical Informatics, 2021. [DOI record](https://doi.org/10.1016/j.jbi.2021.103865)

Thakur and colleagues, “BEIR: A Heterogenous Benchmark for Zero Shot Evaluation of Information Retrieval Models,” NeurIPS Datasets and Benchmarks, 2021. [Paper](https://arxiv.org/abs/2104.08663)

The project combines established retrieval components and follows known query reformulation ideas. Its inspectable controller and session scoped follow up context are implementation and teaching contributions, not a new retrieval algorithm or a claim of state of the art relevance.

## Team and AI use

Raghav Garg built the original project, including the Python indexing and retrieval pipeline, controller and evaluation foundation. Aryan Rastogi tested the project, changed the model provider from Gemini to Groq, built and redesigned the Streamlit app, and led the audit and integration work. Codex assisted with debugging, the sorted postings intersection, rate limit handling, evaluation checks and documentation. Groq generated Boolean query proposals only. Search logic, controller decisions, ranking and evaluation run in project code.

## Submission links

Report Google Doc: Add the link here

Demo video: Add the YouTube or Google Drive link here

## Sources

BEIR paper: https://arxiv.org/abs/2104.08663

TREC COVID paper: https://doi.org/10.1016/j.jbi.2021.103865
