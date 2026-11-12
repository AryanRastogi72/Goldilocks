import math
import unittest

from evaluation.metrics import compute_ndcg_at_k, compute_recall_at_k


class MetricsTests(unittest.TestCase):
    def test_ndcg_linear_gain_ignores_negative_grade(self):
        topic_qrels = {"a": 2, "b": 1, "c": 0, "d": -1, "e": 2}
        ranked = ["d", "a", "c", "b"]  # judged -1 at rank 1, then grade 2, 0, 1
        actual = 0 / math.log2(2) + 2 / math.log2(3) + 0 / math.log2(4) + 1 / math.log2(5)
        ideal = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(compute_ndcg_at_k(ranked, topic_qrels, k=10), actual / ideal)

    def test_recall_counts_only_positive_grades(self):
        relevant = {"a", "b", "e"}  # grade > 0 in the topic above; d (-1) and c (0) are excluded
        self.assertAlmostEqual(compute_recall_at_k(["d", "a", "c", "b"], relevant, k=2), 1 / 3)
        self.assertAlmostEqual(compute_recall_at_k(["d", "a", "c", "b"], relevant, k=4), 2 / 3)


if __name__ == "__main__":
    unittest.main()
