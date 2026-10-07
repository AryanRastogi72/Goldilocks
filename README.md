# Goldilocks

Goldilocks is an information retrieval demo for the TREC COVID collection. It searches 171,332 indexed documents with Boolean queries, uses collection statistics to steer result size, then ranks matching documents with tf idf cosine similarity.

The controller makes retrieval decisions from document frequency, estimated bounds, and actual hit counts. Groq proposes query terms only. Search also works in offline mode without an API call.

## Try the live app

[Open Goldilocks](https://goldilocks-hackathon.streamlit.app/)

The app owner has configured the Groq API key as a private app secret. Visitors do not need to enter an API key. Choose Groq mode to use query proposals, or choose Offline keywords to search without a model call.

## Project contents

* `app.py` is the Streamlit interface with Search, Evaluation, and How it works tabs.
* `engine` contains Boolean parsing, postings search, the controller, the Groq client, and the ranker.
* `pipeline` builds the index from TREC COVID source data.
* `index_out` is the prebuilt index used by the live app.
* `evaluation` contains topic results, metrics, charts, and the evaluation runner.
* `config.json` contains the model and index settings.
* `requirements.txt` lists the pinned Python packages.

Private study notes stay in `docs_private` and are excluded from Git. Raw corpus files, local environments, API keys, and local model response caches are excluded too.

## Run locally on Windows

Open PowerShell in the project folder. Create a virtual environment, install the packages, then start the app:

```powershell
python -m venv .venv
& '.venv\Scripts\Activate.ps1'
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = '1'
streamlit run app.py
```

If PowerShell blocks activation, allow scripts for the current window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& '.venv\Scripts\Activate.ps1'
```

Offline mode works without a key. To test Groq locally, set `GROQ_API_KEY` in the current PowerShell session before starting Streamlit. Do not save the key in this repository. The live app reads its key from Streamlit secrets and does not show a key field to visitors.

## Search and evaluation

Run a Python reference search:

```powershell
$env:PYTHONUTF8 = '1'
& '.venv\Scripts\python.exe' -m engine.main 'covid vaccine pregnant'
```

Run the fifty topic evaluation:

```powershell
& '.venv\Scripts\python.exe' -m evaluation.run_eval
```

The saved offline evaluation reached the target band on 42 of 50 topics. Mean precision was 0.1662, mean recall was 0.0186, and mean P at 10 was 0.2540. The OR baseline mean P at 10 was 0.2880. Low recall is a limitation. The saved topic evaluation predates the Groq integration, so its one shot model baseline is not a Groq result.

## Build the index

The prebuilt index is included. To rebuild it, place the TREC COVID source files in the expected data folder, then run:

```powershell
& '.venv\Scripts\python.exe' -m pipeline.build_index
```

The source corpus and cached topic files are local and ignored by Git. The index contains 171,332 documents and occupies about 208 MB across 38 files.

## Information retrieval concepts

* Tokenization and stemming prepare text for indexing.
* The inverted index maps terms to postings lists.
* Document frequency estimates how many documents contain each term.
* The k gram index supports wildcard term lookup.
* Boolean set operations combine postings for AND, OR, and NOT.
* The controller uses df bounds and actual hit counts to steer result size.
* Tf idf cosine similarity ranks the Boolean matches.
* Precision, recall, and P at 10 compare results with relevance judgments.

## Repository history

GitHub `main` contains a clean project history. Private notes, local data, secrets, and the retired browser implementation are not part of that history.
