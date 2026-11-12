import csv
import json
import os
import tempfile
import unittest

from engine.controller import load_config, run_controller
from engine.evidence import CLASSES, class_of, classify, g_value
from engine.explain import explain, split_sentences
from engine.ranker import apply_evidence_weight, rank_bm25
from engine.search import SearchIndex
from evaluation.score_evidence_labels import score
from tests.support import build_tiny_index

EXAMPLES = {
    "systematic_review_meta_analysis": ("A systematic review of masks", ""),
    "randomized_controlled_trial": ("Remdesivir in adults: a randomized, double-blind trial", ""),
    "observational": ("Outcomes in a retrospective cohort of patients", ""),
    "case_report_series": ("A case of anosmia after infection", ""),
    "modelling_simulation": ("An SEIR model of spread in cities", ""),
    "lab_animal": ("Infection of mice with a coronavirus", ""),
    "opinion_editorial": ("Commentary on testing strategy", ""),
    "other": ("National response plan", "This report describes the plan."),
}


class ClassifierTests(unittest.TestCase):
    def test_one_example_per_class(self):
        for expected, (title, abstract) in EXAMPLES.items():
            self.assertEqual(classify(title, abstract)[0], expected, title)

    def test_no_match_is_other_with_its_own_rule(self):
        self.assertEqual(classify("National response plan", "")[:2], ("other", "NO_MATCH"))

    def test_review_of_randomized_trials_is_a_review(self):
        title = "Masks and respirators: a systematic review of randomized controlled trials"
        self.assertEqual(classify(title, "")[:2], ("systematic_review_meta_analysis", "T_REVIEW"))

    def test_title_rule_beats_abstract_rule(self):
        self.assertEqual(classify("An editorial on masks", "We conducted a meta-analysis.")[0], "opinion_editorial")

    def test_abstract_rule_fires_when_the_title_has_no_cue(self):
        class_id, rule_id, matched = classify("Drug X for fever", "Patients were randomly assigned to drug or placebo.")
        self.assertEqual((class_id, rule_id), ("randomized_controlled_trial", "A_RCT"))
        self.assertEqual(matched, "were randomly assigned")

    def test_g_values_are_between_0_and_1(self):
        config = load_config()
        for class_id in CLASSES:
            self.assertTrue(0 <= g_value(class_id, config) <= 1)

    def test_unknown_class_raises(self):
        with self.assertRaises(ValueError):
            g_value("not_a_class", load_config())


class NetScoreTests(unittest.TestCase):
    def test_zero_weight_keeps_the_order(self):
        ranked = [(3.0, 5), (2.0, 1), (1.0, 9)]
        self.assertEqual(apply_evidence_weight(ranked, 0, lambda doc_id: 1.0), ranked)

    def test_weight_uses_the_normalised_relevance(self):
        ranked = [(10.0, 1), (9.0, 2)]
        g = {1: 0.0, 2: 1.0}
        net = apply_evidence_weight(ranked, 0.5, g.get)
        self.assertEqual([doc_id for _, doc_id in net], [2, 1])
        self.assertAlmostEqual(net[0][0], 9.0 / 10.0 + 0.5 * 1.0)
        self.assertAlmostEqual(net[1][0], 1.0)


class RealIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = SearchIndex()
        cls.config = load_config()

    def test_default_weight_gives_plain_bm25_order(self):
        ranked, _, _ = run_controller("covid vaccine pregnant", self.index, self.config)
        weighted_config = dict(self.config, evidence_weight=0.0)
        again, _, _ = run_controller("covid vaccine pregnant", self.index, weighted_config)
        self.assertEqual(ranked, again)

    def test_filter_keeps_only_that_class(self):
        ranked, log, _ = run_controller("masks", self.index, self.config, evidence_filter="randomized_controlled_trial")
        self.assertTrue(ranked)
        self.assertTrue(all(class_of(self.index, doc_id) == "randomized_controlled_trial" for _, doc_id in ranked))
        self.assertEqual(log.steps[0]["decision"], "EVIDENCE_FILTER")
        self.assertEqual(log.steps[0]["reason"], "evidence filter: randomized_controlled_trial")

    def test_unknown_filter_raises(self):
        with self.assertRaises(ValueError):
            run_controller("masks", self.index, self.config, evidence_filter="best_studies")

    def test_evidence_weight_can_change_the_order(self):
        base, _, _ = run_controller("masks", self.index, self.config)
        boosted, _, _ = run_controller("masks", self.index, dict(self.config, evidence_weight=5.0))
        self.assertNotEqual([doc_id for _, doc_id in base], [doc_id for _, doc_id in boosted])


class ExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        corpus = {
            "d0": {"title": "Alpha study", "text": "Alpha beta here. We note e.g. gamma. Alpha is key."},
            "d1": {"title": "Other", "text": "Beta only appears here. More text follows."},
            "d2": {"title": "No abstract", "text": ""},
            "d3": {"title": "Filler", "text": "delta epsilon zeta"},
        }
        build_tiny_index(corpus, cls.folder.name)
        cls.index = SearchIndex(cls.folder.name)
        cls.config = dict(load_config(), title_boost=0.5)
        cls.concepts = [{"label": "alpha", "stems": ["alpha", "beta"]}]

    @classmethod
    def tearDownClass(cls):
        cls.index.close()
        cls.folder.cleanup()

    def test_contributions_sum_to_the_bm25_score(self):
        ranked = rank_bm25(["alpha", "beta"], self.index, top_k=10, title_boost=0.5, k1=self.config["bm25_k1"], b=self.config["bm25_b"])
        for score, doc_id in ranked:
            result = explain(doc_id, self.concepts, self.index, self.config)
            self.assertAlmostEqual(result["bm25_score"], score, places=6)
        title_part = explain(0, self.concepts, self.index, self.config)["contributions"][0]["title_boost"]
        self.assertGreater(title_part, 0)

    def test_highlight_offsets_point_at_the_terms(self):
        sentence = explain(0, self.concepts, self.index, self.config)["best_sentence"]
        self.assertTrue(sentence["highlights"])
        for start, end in sentence["highlights"]:
            self.assertIn(sentence["text"][start:end].lower(), {"alpha", "beta"})

    def test_abbreviations_do_not_split_sentences(self):
        self.assertEqual(len(split_sentences("We note e.g. gamma here. Next one starts.")), 2)

    def test_no_abstract_gives_a_reason(self):
        result = explain(2, self.concepts, self.index, self.config)
        self.assertIsNone(result["best_sentence"])
        self.assertEqual(result["best_sentence_missing_reason"], "no abstract")


class ScorerTests(unittest.TestCase):
    def test_hand_computed_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            sample, predictions, counts = (os.path.join(folder, n) for n in ("s.csv", "p.csv", "c.json"))
            rows = [("1", "systematic_review_meta_analysis", "systematic_review_meta_analysis"),
                    ("2", "systematic_review_meta_analysis", "systematic_review_meta_analysis"),
                    ("3", "other", "other"), ("4", "other", "systematic_review_meta_analysis")]
            with open(sample, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["doc_id", "ext_id", "title", "abstract", "human_label"])
                writer.writerows([[doc, "x", "t", "a", human] for doc, _, human in rows])
            with open(predictions, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["doc_id", "predicted_label", "rule_id"])
                writer.writerows([[doc, predicted, "R"] for doc, predicted, _ in rows])
            population = {c: {"population": 0, "sampled": 0} for c in CLASSES}
            population["systematic_review_meta_analysis"]["population"] = 100
            population["other"]["population"] = 300
            with open(counts, "w") as f:
                json.dump({"classes": population}, f)

            count, (weighted, accuracy), (_, unweighted_accuracy) = score(sample, predictions, counts)

        # weights: 100 / 2 = 50 and 300 / 2 = 150; weighted matrix: (sys, sys) = 100, (other, other) = 150, (other, sys) = 150
        review = weighted["systematic_review_meta_analysis"]
        self.assertEqual(count, 4)
        self.assertAlmostEqual(review["precision"], 100 / 100)
        self.assertAlmostEqual(review["recall"], 100 / 250)
        self.assertAlmostEqual(review["f1"], 2 * 1.0 * 0.4 / 1.4)
        self.assertAlmostEqual(weighted["other"]["precision"], 150 / 300)
        self.assertAlmostEqual(weighted["other"]["recall"], 150 / 150)
        self.assertAlmostEqual(accuracy, 250 / 400)
        self.assertAlmostEqual(unweighted_accuracy, 3 / 4)

    def test_unknown_human_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            sample, predictions = os.path.join(folder, "s.csv"), os.path.join(folder, "p.csv")
            with open(sample, "w", newline="") as f:
                csv.writer(f).writerows([["doc_id", "ext_id", "title", "abstract", "human_label"], ["1", "x", "t", "a", "guess"]])
            with open(predictions, "w", newline="") as f:
                csv.writer(f).writerows([["doc_id", "predicted_label", "rule_id"], ["1", "other", "R"]])
            with self.assertRaises(ValueError):
                score(sample, predictions, os.path.join(folder, "none.json"))


class AiLabelTests(unittest.TestCase):
    def test_ai_labels_never_touch_human_label(self):
        from evaluation.score_evidence_labels import ai_sample_rows
        rows = ai_sample_rows()
        self.assertEqual(len(rows), 120)
        self.assertTrue(all(row["human_label"] == "" for row in rows))
        self.assertTrue(all(row["ai_label"] in CLASSES for row in rows))


if __name__ == "__main__":
    unittest.main()
