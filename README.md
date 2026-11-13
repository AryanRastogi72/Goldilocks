# EvidenceLens

An offline, explainable research search agent over TREC-COVID (171,332 papers). It rewrites your question, keeps the result set small enough to read, asks a clarifying question when the question is too broad, carries context across follow-ups, labels the study type of every paper, and shows why each result was ranked where it is. It uses no generative AI at query time: every step is classical information retrieval and every decision is logged.

CSD358 Information Retrieval course project, Track 2 (conversational and agentic search), by Aryan Rastogi and Raghav Garg. It extends our midsem project, Goldilocks (see "Midsem history").

## Run it

Python 3.12 or newer and Node.js are needed. From the project folder:

```
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate on macOS and Linux
pip install -r requirements.txt
python -m pipeline.build_index    # reads data/raw/beir/trec-covid.zip, writes index_out/
python -m pipeline.build_thesaurus
python -m pipeline.build_evidence
cd web && npm install && npm run build && cd ..
uvicorn api.main:app              # open http://localhost:8000
```

The three `pipeline` commands are needed after a fresh clone only for `abstracts.sqlite` (made by `build_index`, not tracked in git because of its size); the git repository otherwise tracks the rest of `index_out/`. The submission ZIP leaves `index_out/` out for size, so run all three there. A rebuilt thesaurus can differ slightly from ours, because word2vec training with several workers is not bit-reproducible.

Command-line smoke test: `python -m engine.main "covid vaccine pregnant"`. Tests: `python -m unittest discover tests`. Web development server: `cd web && npm run dev` (port 5173, proxies `/api` to port 8000). Health check: `http://localhost:8000/api/health`.

## What you see

- **Search** shows results only: one line ("I searched for ... - N papers") and a column of result slips with the study-type stamp and the best sentence of the abstract. "Why this result" shows the score, the per-term BM25 bars and the rule behind the study type.
- **How did I get this?** opens a side sheet with the concepts, the numbered trace of tighten and relax steps against the 20 to 200 band, the Boolean query and earlier turns. A switch there re-ranks the current turn with plain BM25 over the whole collection.
- **How it works** explains the pipeline. **Evaluation** shows the saved results, starting with where the agent loses.

## Architecture

```
question -> conversation rules -> concept extraction (idf, lexicon, index elimination)
         -> Boolean controller over the inverted index (band 20-200: accept / tighten / relax)
              uses: word2vec thesaurus, Rocchio terms, k-gram wildcards, title zone
         -> too broad? clusters of the result set become clarifying choices
         -> BM25 ranking (+ optional evidence score) -> per-result explanation -> web app
```

| Folder | Content |
|---|---|
| `pipeline/` | index, thesaurus and evidence-label builders |
| `engine/` | Boolean parser, controller, ranker, conversation rules, clarification, evidence classifier, explainer |
| `api/` | FastAPI app (`/api/search`, `/api/clarify`, `/api/rerank`, `/api/doc/{id}`, `/api/eval`, `/api/health`) |
| `web/` | React app ("Bench Notebook" design) |
| `evaluation/` | six runs, significance tests, charts, saved results (`summary.md`) |
| `docs/` | brief, formulas, references, viva prep, demo script, report (`docs/report/main.pdf`) |

## IR concepts used

Inverted index with sharded postings; Boolean retrieval with postings intersection ordered by document frequency; k-gram index for wildcards; title-zone postings; index elimination; Okapi BM25 (Lucene idf, k1 = 1.2, b = 0.75); ltc.ltc tf-idf cosine; Rocchio pseudo-relevance feedback (positive part only); a word2vec thesaurus for global query expansion; leader-follower clustering for clarification, with contrast-score labels; rule-based query rewriting for follow-ups; a rule-based study-type classifier with an evidence score g(d) from a published levels-of-evidence ordering; paired randomization tests with Bonferroni correction. The formulas are in `docs/formulas.md`.

## Evaluation, with the trade-offs

Fifty TREC-COVID topics, both the keyword field and the question field, depth 1000, nDCG with linear gain. Full numbers: `evaluation/summary.md`.

| run | nDCG@10 keyword | nDCG@10 question |
|---|---|---|
| R1 BM25, whole collection | 0.6032 | 0.6555 |
| R3 BM25 + Rocchio | 0.5741 | 0.6180 |
| R4 EvidenceLens agent | 0.5897 | 0.4452 |
| R5 agent + evidence weight 0.2 | 0.5856 | 0.4363 |

- The agent keeps 98% (keyword) and 96% (question) of topics inside the 20 to 200 band, against 32% and 12% for the midsem AND baseline.
- It is **tied with BM25 on keyword queries** (p = 0.6737) and **clearly worse on question text** (p = 0.0001). Because the band caps the set at about 200 papers, its Recall@1000 is 0.1217 and 0.0531 against 0.4161 and 0.4543 for BM25.
- Rocchio feedback and the evidence weight did not improve ranking. The controller revision (v2.1) was made after seeing v2.0 on all 50 topics, with no held-out topics, so it is a post-hoc result.
- Our BM25 on the question field gets nDCG@10 0.6555; the BEIR paper reports 0.656 for BM25 on TREC-COVID (Table 2, version 4). Their BM25 setup differs from ours, so we do not claim to reproduce it.
- Study-type classifier: accuracy against hand labels is `TODO(metric)` (the 120-document blind sample in `evaluation/evidence_label_sample.csv` has not been hand-labelled yet).

## Team

- Raghav Garg: the original Goldilocks indexing, retrieval, controller and evaluation.
- Aryan Rastogi: the midsem model-provider switch, the midsem web UI and audit; EvidenceLens product direction and decisions.
- Everything else in EvidenceLens: `TODO(team)` (to be filled by Aryan and Raghav).
- Source code link for the submission: `TODO(link)`.

## Midsem history

Goldilocks, the midsem project, controlled result size with a Boolean query controller and ranked with tf-idf cosine, but relied on a language-model API to propose queries; 14 of 50 topics had no valid proposal in the saved replay. Its state is frozen in the git tag `midsem-final` (commit 37d95aa) and its saved results are in `archive/midsem/evaluation/summary.tsv`: 86.0% of topics inside the band, mean 73.5 hits, P@10 0.4860, nDCG@10 0.4549 and MRR@10 0.6593. Those nDCG numbers use gain 2^g - 1, so they are not comparable with the linear-gain numbers above.

## Data and references

TREC-COVID through BEIR (Roberts et al., Journal of Biomedical Informatics 2021; Thakur et al., 2021). The levels-of-evidence ordering behind g(d) is from Burns, Rohrich and Chung (Plastic and Reconstructive Surgery, 2011). Verification notes for every reference: `docs/references.md`.
