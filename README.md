# Goldilocks

Goldilocks is an information retrieval demo for the TREC COVID collection. It searches 171,332 indexed documents with Boolean queries, uses collection statistics to steer the result size, and ranks matching documents with tf idf cosine similarity.

The controller is responsible for retrieval decisions. Gemini can suggest query terms, but document frequency, estimated result bounds, and actual hit counts drive decisions to accept, tighten, relax, or stop. The app also runs without an API key by using an offline keyword proposal.

## What is included

* `app.py` is the Streamlit interface with Search, Evaluation, and How it works tabs.
* `engine` contains the Boolean parser, postings search, controller, Gemini client, and tf idf ranker.
* `pipeline` downloads or builds the corpus index.
* `index_out` is the prebuilt index used by the app.
* `evaluation` contains the saved topic results, summary, charts, and evaluation runner.
* `config.json` contains the Gemini model and index settings.
* `requirements.txt` pins the Python packages used by the app.

Private notes in `docs_private` remain available in the local project and are excluded from Git. Raw corpus files, local environments, and API keys are excluded too.

## Run locally on Windows

Use PowerShell from this folder. Create and activate a virtual environment, install the packages, then start the app:

```powershell
python -m venv .venv
& '.venv\Scripts\Activate.ps1'
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = '1'
streamlit run app.py
```

If PowerShell blocks activation, allow scripts for this window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& '.venv\Scripts\Activate.ps1'
```

Open the local address printed by Streamlit. Offline mode works without a key. In Gemini mode, enter a key in the password field. The app keeps the visitor key in session state. For a deployment wide key, use Streamlit secrets rather than a committed file.

## Search and evaluation

Run the Python reference search from PowerShell:

```powershell
$env:PYTHONUTF8 = '1'
& '.venv\Scripts\python.exe' -m engine.main 'covid vaccine pregnant'
```

Run the fifty topic evaluation:

```powershell
& '.venv\Scripts\python.exe' -m evaluation.run_eval
```

The saved offline evaluation reached the target band on 42 of 50 topics. Mean precision was 0.1662, mean recall was 0.0186, and mean P at 10 was 0.2540. The OR baseline mean P at 10 was 0.2880. The low recall is a limitation. Gemini evaluation is not available because a live Gemini proposal has not succeeded in the current environment.

## Build the index

The prebuilt index is included. To rebuild it, first place the TREC COVID source files in the expected data folder, then run:

```powershell
& '.venv\Scripts\python.exe' -m pipeline.build_index
```

The raw corpus and cached topic files are local and ignored by Git. The index contains 171,332 documents and occupies about 208 MB across 38 files.

## Deploy on Streamlit Community Cloud

1. Use a clean public Git repository with the app source, `requirements.txt`, configuration, evaluation outputs, and `index_out`.
2. Keep `.env`, Streamlit secrets, `docs_private`, raw source data, and local environments out of the repository.
3. Select `app.py` as the app entry point.
4. Select Python 3.13 to match the locally tested environment.
5. Deploy with no secret for offline search, or add a Gemini key through Streamlit secrets.
6. Test startup, offline search, and memory use after deployment.

Each index file is below GitHub's individual file size limit. The full index is about 208 MB, so allow time for upload and app startup.

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

The original working history contained private notes. The GitHub `main` branch now points to a clean root history that excludes those notes. This local checkout still has its earlier history, so publish future changes from the clean history rather than pushing from the old local `.git` history.
