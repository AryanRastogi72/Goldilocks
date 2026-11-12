import os
import random
import tempfile
import unittest
from unittest import mock

from evaluation.metrics import compute_map_at_k, compute_set_metrics
from evaluation.significance import randomization_p_value
from evaluation.validate_results import check_run, read_trec


class MetricTests(unittest.TestCase):
    def test_map_at_k_by_hand(self):
        # relevant: a, c, e (3 documents); ranking: a (relevant), b, c (relevant), d, f; e is never retrieved
        ranked = ["a", "b", "c", "d", "f"]
        precision_at_rank_1 = 1 / 1
        precision_at_rank_3 = 2 / 3
        expected = (precision_at_rank_1 + precision_at_rank_3) / 3
        self.assertAlmostEqual(compute_map_at_k(ranked, {"a", "c", "e"}, k=100), expected)

    def test_map_cuts_at_k(self):
        self.assertAlmostEqual(compute_map_at_k(["x", "a"], {"a"}, k=1), 0.0)

    def test_set_metrics_by_hand(self):
        scores = compute_set_metrics({"a", "b", "c", "d"}, {"a", "b", "z"})
        self.assertAlmostEqual(scores["set_precision"], 2 / 4)
        self.assertAlmostEqual(scores["set_recall"], 2 / 3)
        self.assertAlmostEqual(scores["set_f1"], 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))

    def test_empty_set_gives_zeros(self):
        self.assertEqual(compute_set_metrics(set(), {"a"}), {"set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0})


class SignificanceTests(unittest.TestCase):
    def test_constant_improvement_is_significant(self):
        p = randomization_p_value([0.1] * 20, random.Random(42), permutations=2000)
        self.assertLess(p, 0.01)

    def test_symmetric_differences_are_not_significant(self):
        p = randomization_p_value([0.2, -0.2, 0.1, -0.1, 0.3, -0.3], random.Random(42), permutations=2000)
        self.assertGreater(p, 0.9)

    def test_same_seed_gives_the_same_p(self):
        differences = [0.3, -0.1, 0.2, 0.0, 0.05, 0.4, -0.2, 0.1]
        first = randomization_p_value(differences, random.Random(42), permutations=500)
        again = randomization_p_value(differences, random.Random(42), permutations=500)
        self.assertEqual(first, again)


class ValidatorTests(unittest.TestCase):
    def run_check(self, trec_lines, record):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "run.trec")
            with open(path, "w") as f:
                f.write("\n".join(trec_lines) + "\n")
            problems = []
            qrels = {"1": {"a": 2, "b": 0}}
            with mock.patch("evaluation.validate_results.trec_path", return_value=path), \
                    mock.patch("evaluation.validate_results.EXPECTED_TOPICS", 1):
                check_run("RX", "text", {"per_topic": {"1": record}}, qrels, problems)
        return problems

    def good_record(self):
        return {"p_at_10": 0.1, "ndcg_at_10": 1.0, "mrr_at_10": 1.0, "map_at_100": 1.0,
                "recall_at_100": 1.0, "recall_at_1000": 1.0, "retrieved": 2}

    def test_matching_results_pass(self):
        self.assertEqual(self.run_check(["1 Q0 a 1 2.0 RX", "1 Q0 b 2 1.0 RX"], self.good_record()), [])

    def test_a_changed_metric_is_reported(self):
        record = dict(self.good_record(), ndcg_at_10=0.9)
        problems = self.run_check(["1 Q0 a 1 2.0 RX", "1 Q0 b 2 1.0 RX"], record)
        self.assertTrue(any("ndcg_at_10" in problem for problem in problems))

    def test_duplicate_documents_are_reported(self):
        problems = self.run_check(["1 Q0 a 1 2.0 RX", "1 Q0 a 2 1.0 RX"], self.good_record())
        self.assertTrue(any("duplicate" in problem for problem in problems))

    def test_malformed_line_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.trec")
            with open(path, "w") as f:
                f.write("1 a 1 2.0\n")
            with self.assertRaises(ValueError):
                read_trec(path)


if __name__ == "__main__":
    unittest.main()
