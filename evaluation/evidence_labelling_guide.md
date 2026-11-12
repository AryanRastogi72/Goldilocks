# Evidence labelling guide

Fill the `human_label` column of `evaluation/evidence_label_sample.csv` with exactly one class id per row, using the title and abstract only. Do not open `evidence_label_predictions.csv` until you have finished: it holds the classifier's guesses and would bias you.

**Who labels what:** Aryan labels rows 1 to 60, Raghav labels rows 61 to 120 (row 1 is the first data row under the header). Label what the paper *is*, not what it is about. If two classes seem to fit, choose the one the authors' main design belongs to; if you cannot tell, use `other`.

| class id | what it is | one cue | one counter-example |
|---|---|---|---|
| `systematic_review_meta_analysis` | Collects and appraises many studies with a stated search or pools their results | "we searched PubMed and Embase and pooled ..." | A narrative overview with no search method (use `other` or `opinion_editorial`) |
| `randomized_controlled_trial` | Participants are randomly assigned to interventions and outcomes compared | "patients were randomly assigned to ..." | A review of trials (that is a systematic review); a single-arm trial with no randomisation (`other`) |
| `observational` | Follows or compares groups without assigning the exposure: cohort, case-control, cross-sectional, registry | "retrospective cohort of 1,200 patients" | A report of one or a few patients (`case_report_series`) |
| `case_report_series` | Describes one patient or a small series of patients | "we report a 34-year-old woman who ..." | A large cohort that is called a "series" but analysed statistically (`observational`) |
| `modelling_simulation` | The result comes from a mathematical or computer model, not from collected patient data | "an SEIR model projects ..." | A statistical analysis of case counts without a mechanistic model (`observational`) |
| `lab_animal` | Laboratory, cell-culture or animal experiments, or biochemical and structural work | "in vitro" assay, "infected mice" | A clinical study that measures a lab marker in patients (`observational`) |
| `opinion_editorial` | Editorial, commentary, letter, viewpoint or perspective: argument or opinion rather than new data | "In this commentary we argue ..." | A guideline built from a systematic search (`systematic_review_meta_analysis` or `other`) |
| `other` | Anything else, including guidelines, surveys, descriptive reports, policy documents, data resources | "this report describes the national response plan" | A paper that fits one of the classes above |

When finished, run `python -m evaluation.score_evidence_labels`. It checks that every label is one of the ids above, then reports per-class precision, recall, F1 and accuracy.
