import math
import tempfile
import unittest

from engine.search import SearchIndex
from engine.ranker import rank_bm25
from tests.support import build_tiny_index

CORPUS = {
    "doc_a": {"title": "cat", "text": "dog dog"},         # indexed tokens: cat dog dog (3)
    "doc_b": {"title": "bird", "text": "cat cat cat bird"},  # bird cat cat cat bird (5)
    "doc_c": {"title": "fish", "text": "fish fish"},      # fish fish fish (3)
}
K1 = 1.2
B = 0.75


class Bm25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        build_tiny_index(CORPUS, cls.folder.name)
        cls.index = SearchIndex(cls.folder.name)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_scores_match_hand_calculation(self):
        num_docs = 3
        df_cat = 2
        avg_dl = 3.67  # 11 tokens / 3 docs, stored rounded to 2 decimals
        idf_cat = math.log(1 + (num_docs - df_cat + 0.5) / (df_cat + 0.5))
        length_norm_a = 1 - B + B * 3 / avg_dl
        length_norm_b = 1 - B + B * 5 / avg_dl
        score_a = idf_cat * 1 * (K1 + 1) / (1 + K1 * length_norm_a)
        score_b = idf_cat * 3 * (K1 + 1) / (3 + K1 * length_norm_b)

        ranked = rank_bm25(["cat"], self.index, top_k=10)

        self.assertEqual([doc_id for _, doc_id in ranked], [1, 0])
        self.assertAlmostEqual(ranked[0][0], score_b)
        self.assertAlmostEqual(ranked[1][0], score_a)

    def test_title_boost_changes_order(self):
        without_boost = rank_bm25(["cat"], self.index, top_k=10)
        with_boost = rank_bm25(["cat"], self.index, top_k=10, title_boost=1.0)
        self.assertEqual(without_boost[0][1], 1)
        self.assertEqual(with_boost[0][1], 0)
        idf_cat = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        plain_score_a = dict((doc_id, score) for score, doc_id in without_boost)[0]
        self.assertAlmostEqual(with_boost[0][0], plain_score_a + idf_cat)

    def test_doc_ids_restricts_candidates(self):
        ranked = rank_bm25(["cat"], self.index, top_k=10, doc_ids={0})
        self.assertEqual([doc_id for _, doc_id in ranked], [0])

    def test_unknown_term_gives_empty_result(self):
        self.assertEqual(rank_bm25(["zebra"], self.index), [])

    def test_ties_break_by_smaller_doc_id(self):
        ranked = rank_bm25(["cat"], self.index, top_k=10, k1=0.0)  # k1=0 makes both docs score idf
        self.assertAlmostEqual(ranked[0][0], ranked[1][0])
        self.assertEqual([doc_id for _, doc_id in ranked], [0, 1])


if __name__ == "__main__":
    unittest.main()
