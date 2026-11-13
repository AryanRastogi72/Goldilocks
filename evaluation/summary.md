# Evaluation summary

R5 evidence_weight 0.2 was chosen by the designer before any R5 result existed; nothing else was tuned.

All runs use depth 1000, 50 topics, relevant = grade above 0, nDCG with linear gain. Run files: `evaluation/runs/*.trec`; recomputed by `python -m evaluation.validate_results`.

## Ranking metrics (mean over 50 topics)

| run | field | P@10 | nDCG@10 | MRR@10 | MAP@100 | Recall@100 | Recall@1000 | latency s |
|---|---|---|---|---|---|---|---|---|
| R0 | metadata.query | 0.5200 | 0.4663 | 0.7397 | 0.0593 | 0.0873 | 0.2108 | 0.143 |
| R0 | text | 0.1360 | 0.1284 | 0.2695 | 0.0065 | 0.0081 | 0.0091 | 0.054 |
| R1 | metadata.query | 0.6660 | 0.6032 | 0.8287 | 0.0859 | 0.1176 | 0.4161 | 0.151 |
| R1 | text | 0.6860 | 0.6555 | 0.8875 | 0.0904 | 0.1284 | 0.4543 | 0.168 |
| R2 | metadata.query | 0.3660 | 0.3343 | 0.5340 | 0.0369 | 0.0656 | 0.3538 | 0.084 |
| R2 | text | 0.3720 | 0.3392 | 0.5454 | 0.0353 | 0.0689 | 0.3681 | 0.180 |
| R3 | metadata.query | 0.6380 | 0.5741 | 0.7822 | 0.0813 | 0.1158 | 0.4488 | 0.208 |
| R3 | text | 0.6680 | 0.6180 | 0.8037 | 0.0871 | 0.1269 | 0.4824 | 0.415 |
| R4 | metadata.query | 0.6480 | 0.5897 | 0.7857 | 0.0721 | 0.0990 | 0.1217 | 0.106 |
| R4 | text | 0.4940 | 0.4452 | 0.6929 | 0.0280 | 0.0452 | 0.0531 | 0.061 |
| R5 | metadata.query | 0.6460 | 0.5856 | 0.7619 | 0.0725 | 0.0996 | 0.1217 | 0.085 |
| R5 | text | 0.4800 | 0.4363 | 0.6836 | 0.0278 | 0.0450 | 0.0531 | 0.061 |

Note on Recall@1000: R4 and R5 return one accepted set, and the band caps that set at about 200 documents (mean 128 hits on the question text), while the 50 topics average 493.5 judged-relevant documents (minimum 111, maximum 1266; computed from the qrels). Their Recall@1000 can therefore never reach that of a ranking run: R4 0.1217 / 0.0531 against R1 0.4161 / 0.4543 (keyword query / question text). The agent trades recall for a reading-sized result set.

## Set metrics for set runs (R0, R4, R5)

| run | field | mean hits | set precision | set recall | set F1 | in band |
|---|---|---|---|---|---|---|
| R0 | metadata.query | 517.5 | 0.3324 | 0.2224 | 0.2116 | 32% |
| R0 | text | 10.1 | 0.1627 | 0.0091 | 0.0145 | 12% |
| R4 | metadata.query | 107.4 | 0.4656 | 0.1217 | 0.1732 | 98% |
| R4 | text | 128.2 | 0.2639 | 0.0535 | 0.0776 | 96% |
| R5 | metadata.query | 107.4 | 0.4656 | 0.1217 | 0.1732 | 98% |
| R5 | text | 128.2 | 0.2639 | 0.0535 | 0.0776 | 96% |

## Significance (paired randomization tests)

Bonferroni threshold for 20 tests: 0.05 / 20 = 0.0025.

| field | metric | pair | mean difference (first minus second) | p | first wins | second wins | tied | significant after Bonferroni |
|---|---|---|---|---|---|---|---|---|
| metadata.query | ndcg_at_10 | R1 vs R2 | +0.2688 | 0.0001 | 42 | 6 | 2 | yes |
| metadata.query | p_at_10 | R1 vs R2 | +0.3000 | 0.0001 | 44 | 3 | 3 | yes |
| metadata.query | ndcg_at_10 | R1 vs R3 | +0.0290 | 0.1279 | 25 | 21 | 4 | no |
| metadata.query | p_at_10 | R1 vs R3 | +0.0280 | 0.2200 | 17 | 12 | 21 | no |
| metadata.query | ndcg_at_10 | R1 vs R4 | +0.0134 | 0.6737 | 23 | 23 | 4 | no |
| metadata.query | p_at_10 | R1 vs R4 | +0.0180 | 0.6083 | 23 | 15 | 12 | no |
| metadata.query | ndcg_at_10 | R4 vs R5 | +0.0042 | 0.6034 | 15 | 16 | 19 | no |
| metadata.query | p_at_10 | R4 vs R5 | +0.0020 | 1.0000 | 6 | 7 | 37 | no |
| metadata.query | ndcg_at_10 | R0 vs R4 | -0.1234 | 0.0019 | 16 | 32 | 2 | yes |
| metadata.query | p_at_10 | R0 vs R4 | -0.1280 | 0.0011 | 12 | 30 | 8 | yes |
| text | ndcg_at_10 | R1 vs R2 | +0.3163 | 0.0001 | 43 | 7 | 0 | yes |
| text | p_at_10 | R1 vs R2 | +0.3140 | 0.0001 | 42 | 5 | 3 | yes |
| text | ndcg_at_10 | R1 vs R3 | +0.0375 | 0.0351 | 29 | 18 | 3 | no |
| text | p_at_10 | R1 vs R3 | +0.0180 | 0.3981 | 18 | 16 | 16 | no |
| text | ndcg_at_10 | R1 vs R4 | +0.2103 | 0.0001 | 38 | 12 | 0 | yes |
| text | p_at_10 | R1 vs R4 | +0.1920 | 0.0001 | 28 | 10 | 12 | yes |
| text | ndcg_at_10 | R4 vs R5 | +0.0089 | 0.1418 | 16 | 13 | 21 | no |
| text | p_at_10 | R4 vs R5 | +0.0140 | 0.2804 | 9 | 6 | 35 | no |
| text | ndcg_at_10 | R0 vs R4 | -0.3168 | 0.0001 | 2 | 44 | 4 | yes |
| text | p_at_10 | R0 vs R4 | -0.3580 | 0.0001 | 2 | 43 | 5 | yes |

## BEIR comparison (question text field, nDCG@10)

BEIR paper (Thakur et al., arXiv 2104.08663, version v4 (21 Oct 2021)), Table 2 (nDCG@10, zero-shot performance): BM25 on TREC-COVID nDCG@10 = 0.656. Our R1 (BM25, question text field) = 0.6555; difference -0.0005.

The gap is 0.0005 nDCG@10 points, which is within what small implementation differences can produce; with different k1, b, tokenizer and field handling an exact match would be a coincidence, so we do not claim that we reproduce BEIR's pipeline.

Differences that could contribute (none was tested one by one):
- BEIR's BM25 is Anserini with Lucene defaults k1 = 0.9, b = 0.4; ours uses k1 = 1.2, b = 0.75 and a Lucene-style idf.
- BEIR indexes title and text as separate fields; we index one stream of title plus abstract tokens with a single length.
- Tokenizer and stemming differ (our regex tokenizer with Porter stemming and no stopword removal while indexing; query stopwords are removed on our side).
- Lucene may store document length in a lossy encoded norm; we use exact token counts (this detail comes from general knowledge, not from the BEIR paper, and was not checked).
- The BEIR corpus version and judgement handling may differ from the files we use (our run uses the same 171,332 documents and 50 topics).

## Evidence classifier accuracy (human labels)

TODO(metric): the blind sample `evaluation/evidence_label_sample.csv` has not been labelled by a human yet; after labelling, `python -m evaluation.score_evidence_labels` saves `evaluation/evidence_scores.json`.

## Where EvidenceLens wins and loses

R4 is the EvidenceLens agent (controller v2.1) and R1 is plain BM25 on the full collection.

### keyword query (`metadata.query`)

Comparisons:
- R1 vs R2, ndcg_at_10: R1 is higher by 0.2688 (p = 0.0001, R1 wins 42, R2 wins 6, tied 2); significant after Bonferroni correction.
- R1 vs R2, p_at_10: R1 is higher by 0.3000 (p = 0.0001, R1 wins 44, R2 wins 3, tied 3); significant after Bonferroni correction.
- R1 vs R3, ndcg_at_10: R1 is higher by 0.0290 (p = 0.1279, R1 wins 25, R3 wins 21, tied 4); not significant at 0.05.
- R1 vs R3, p_at_10: R1 is higher by 0.0280 (p = 0.2200, R1 wins 17, R3 wins 12, tied 21); not significant at 0.05.
- R1 vs R4, ndcg_at_10: R1 is higher by 0.0134 (p = 0.6737, R1 wins 23, R4 wins 23, tied 4); not significant at 0.05.
- R1 vs R4, p_at_10: R1 is higher by 0.0180 (p = 0.6083, R1 wins 23, R4 wins 15, tied 12); not significant at 0.05.
- R4 vs R5, ndcg_at_10: R4 is higher by 0.0042 (p = 0.6034, R4 wins 15, R5 wins 16, tied 19); not significant at 0.05.
- R4 vs R5, p_at_10: R4 is higher by 0.0020 (p = 1.0000, R4 wins 6, R5 wins 7, tied 37); not significant at 0.05.
- R0 vs R4, ndcg_at_10: R4 is higher by 0.1234 (p = 0.0019, R0 wins 16, R4 wins 32, tied 2); significant after Bonferroni correction.
- R0 vs R4, p_at_10: R4 is higher by 0.1280 (p = 0.0011, R0 wins 12, R4 wins 30, tied 8); significant after Bonferroni correction.

Topics where R4 beats R1 most (nDCG@10):

| topic | text | R1 | R4 | difference | R4 final Boolean query | R4 hits |
|---|---|---|---|---|---|---|
| 33 | coronavirus vaccine candidates | 0.152 | 0.843 | +0.691 | `(candid AND vaccin AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru) AND title:candid)` | 70 |
| 40 | coronavirus mutations | 0.518 | 1.000 | +0.482 | `(mutat AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru) AND title:mutat)` | 193 |
| 12 | coronavirus quarantine | 0.084 | 0.497 | +0.413 | `(quarantin AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru) AND title:quarantin AND hubei)` | 22 |
| 4 | how do people die from the coronavirus | 0.032 | 0.254 | +0.223 | `(die AND peopl AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru) AND title:peopl)` | 22 |
| 34 | coronavirus recovery | 0.078 | 0.290 | +0.211 | `(recoveri AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru) AND title:recoveri)` | 271 |

Topics where R4 loses to R1 most (nDCG@10):

| topic | text | R1 | R4 | difference | R4 final Boolean query | R4 hits |
|---|---|---|---|---|---|---|
| 19 | what alcohol sanitizer kills coronavirus | 0.574 | 0.139 | -0.435 | `((sanit OR hygien OR soap OR disinfect OR soak OR soar OR soa OR soaic OR soapberri) AND (kill OR sicken OR sick OR sickl OR sickest OR sicker OR sickb))` | 73 |
| 42 | Vitamin D and COVID-19 | 1.000 | 0.586 | -0.414 | `(vitamin AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru) AND title:vitamin)` | 157 |
| 21 | coronavirus mortality | 0.961 | 0.572 | -0.389 | `((mortal OR death OR fatal) AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru) AND title:fatal AND cfr)` | 77 |
| 49 | post-infection COVID-19 immunity | 0.429 | 0.066 | -0.363 | `(post AND immun AND infect AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru))` | 169 |
| 44 | impact of masks on coronavirus transmission | 0.865 | 0.545 | -0.320 | `((mask OR respir OR n95) AND transmiss AND impact)` | 80 |

### question text (`text`)

Comparisons:
- R1 vs R2, ndcg_at_10: R1 is higher by 0.3163 (p = 0.0001, R1 wins 43, R2 wins 7, tied 0); significant after Bonferroni correction.
- R1 vs R2, p_at_10: R1 is higher by 0.3140 (p = 0.0001, R1 wins 42, R2 wins 5, tied 3); significant after Bonferroni correction.
- R1 vs R3, ndcg_at_10: R1 is higher by 0.0375 (p = 0.0351, R1 wins 29, R3 wins 18, tied 3); significant at 0.05 but not after Bonferroni correction.
- R1 vs R3, p_at_10: R1 is higher by 0.0180 (p = 0.3981, R1 wins 18, R3 wins 16, tied 16); not significant at 0.05.
- R1 vs R4, ndcg_at_10: R1 is higher by 0.2103 (p = 0.0001, R1 wins 38, R4 wins 12, tied 0); significant after Bonferroni correction.
- R1 vs R4, p_at_10: R1 is higher by 0.1920 (p = 0.0001, R1 wins 28, R4 wins 10, tied 12); significant after Bonferroni correction.
- R4 vs R5, ndcg_at_10: R4 is higher by 0.0089 (p = 0.1418, R4 wins 16, R5 wins 13, tied 21); not significant at 0.05.
- R4 vs R5, p_at_10: R4 is higher by 0.0140 (p = 0.2804, R4 wins 9, R5 wins 6, tied 35); not significant at 0.05.
- R0 vs R4, ndcg_at_10: R4 is higher by 0.3168 (p = 0.0001, R0 wins 2, R4 wins 44, tied 4); significant after Bonferroni correction.
- R0 vs R4, p_at_10: R4 is higher by 0.3580 (p = 0.0001, R0 wins 2, R4 wins 43, tied 5); significant after Bonferroni correction.

Topics where R4 beats R1 most (nDCG@10):

| topic | text | R1 | R4 | difference | R4 final Boolean query | R4 hits |
|---|---|---|---|---|---|---|
| 37 | What is the result of phylogenetic analysis of SARS-CoV-2 genome sequence? | 0.527 | 0.945 | +0.418 | `(phylogenet AND genom AND sequenc AND analysi AND result AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru))` | 122 |
| 43 | How has the COVID-19 pandemic impacted violence in society, including violent crimes? | 0.645 | 0.927 | +0.281 | `((violent OR genocid OR wildfir OR crime) AND (crime OR secular OR violent OR cybercrim) AND (violenc OR maltreat OR abus OR racism))` | 24 |
| 1 | what is the origin of COVID-19 | 0.502 | 0.760 | +0.258 | `(origin AND (covid OR covid19 OR sarscov2 OR ncov OR cov2 OR coronaviru) AND title:origin)` | 177 |
| 25 | which biomarkers predict the severe clinical course of 2019-nCOV infection? | 0.469 | 0.584 | +0.115 | `(biomark AND cours AND predict)` | 27 |
| 21 | what are the mortality rates overall and in specific populations | 0.484 | 0.596 | +0.112 | `(overal AND popul AND rate AND specif)` | 107 |

Topics where R4 loses to R1 most (nDCG@10):

| topic | text | R1 | R4 | difference | R4 final Boolean query | R4 hits |
|---|---|---|---|---|---|---|
| 48 | what are the benefits and risks of re-opening schools in the midst of the COVID-19 pandemic? | 0.779 | 0.000 | -0.779 | `((midst OR amidst OR era OR aftermath) AND (benefit OR outweigh OR reap OR harm OR reaction OR real OR reactiv OR reach OR reason) AND open)` | 28 |
| 50 | what is known about an mRNA vaccine for the SARS-CoV-2 virus? | 0.748 | 0.000 | -0.748 | `(mrna AND known AND vaccin)` | 23 |
| 23 | what kinds of complications related to COVID-19 are associated with hypertension? | 0.734 | 0.000 | -0.734 | `((kind OR recip OR endless OR guis OR guidelin OR guid OR guidanc OR guinea OR guillain) AND (hypertens OR dyslipidaemia OR mellitu OR diabet) AND (complic OR sequela OR sequala OR morbid))` | 121 |
| 24 | what kinds of complications related to COVID-19 are associated with diabetes | 0.792 | 0.104 | -0.688 | `((kind OR recip OR endless OR guis OR guidelin OR guid OR guidanc OR guinea OR guillain) AND (diabet OR mellitu OR hypertens OR obes) AND (complic OR sequela OR sequala OR morbid))` | 156 |
| 40 | What are the observed mutations in the SARS-CoV-2 genome and how often do the mutations occur? | 1.000 | 0.315 | -0.685 | `(mutat AND often AND genom)` | 51 |
