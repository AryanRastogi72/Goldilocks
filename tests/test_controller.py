import ast
import glob
import json
import os
import tempfile
import unittest

from engine.boolean_parser import ast_to_string, parse
from engine.controller import load_config, run_controller
from engine.expansion import prf_terms
from engine.query_understanding import understand
from engine.search import SearchIndex
from tests.support import build_tiny_index

NETWORK_MODULES = {"requests", "urllib", "urllib.request", "http", "http.client", "socket"}


def decisions(log):
    return [step["decision"] for step in log.steps]


class RealIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = SearchIndex()
        cls.config = load_config()

    def test_quoted_words_are_required(self):
        concepts = understand('"remdesivir" children', self.index, self.config)
        by_label = {c["label"]: c for c in concepts}
        self.assertTrue(by_label["remdesivir"]["required"])
        self.assertFalse(by_label["children"]["required"])

    def test_lexicon_group_becomes_one_concept(self):
        concepts = understand("covid-19 masks", self.index, self.config)
        labels = [c["label"] for c in concepts]
        self.assertEqual(sorted(labels), ["covid19", "face_mask"])
        mask = next(c for c in concepts if c["label"] == "face_mask")
        self.assertIn("mask", mask["stems"])
        self.assertIn("n95", mask["stems"])
        self.assertEqual(mask["source"], "lexicon")

    def test_common_concept_is_eliminated_and_order_is_by_idf(self):
        concepts = understand("covid vaccine pregnant", self.index, self.config)
        self.assertEqual([c["label"] for c in concepts], ["pregnancy", "vaccine", "covid19"])
        self.assertEqual([c["eliminated"] for c in concepts], [False, False, True])

    def test_single_common_concept_is_kept(self):
        concepts = understand("covid", self.index, self.config)
        self.assertFalse(concepts[0]["eliminated"])

    def test_controller_reaches_band_on_three_queries(self):
        low, high = self.config["target_band_low"], self.config["target_band_high"]
        for query in ["covid vaccine pregnant", "masks", "what is the origin of covid-19"]:
            ranked, log, final_query = run_controller(query, self.index, self.config)
            self.assertEqual(log.steps[-1]["decision"], "ACCEPT", query)
            self.assertTrue(low <= log.steps[-1]["actual_hits"] <= high, query)
            self.assertEqual(len(ranked), self.config["top_k"])

    def test_tighten_fires(self):
        _, log, _ = run_controller("covid vaccine pregnant", self.index, self.config)
        self.assertIn("TIGHTEN_ADD_CONCEPT", decisions(log))

    def test_relax_fires(self):
        query = "coronavirus origin transmission asymptomatic children pregnant remdesivir"
        _, log, _ = run_controller(query, self.index, self.config)
        self.assertTrue(any(d.startswith("RELAX") for d in decisions(log)), decisions(log))

    def test_every_step_names_its_source(self):
        _, log, _ = run_controller("masks", self.index, self.config)
        allowed = {"query", "lexicon", "thesaurus", "prf", "kgram", "rule"}
        self.assertTrue(all(step["source"] in allowed for step in log.steps))
        json.dumps(log.to_dict())


class TinyIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        corpus = {f"top{i}": {"title": "alpha", "text": "alpha beta beta gamma"} for i in range(6)}
        corpus.update({f"low{i}": {"title": "delta", "text": "delta epsilon"} for i in range(5)})
        corpus["rare"] = {"title": "alpha", "text": "alpha zeta"}
        build_tiny_index(corpus, cls.folder.name)
        cls.index = SearchIndex(cls.folder.name)

    @classmethod
    def tearDownClass(cls):
        cls.index.close()
        cls.folder.cleanup()

    def test_prf_picks_frequent_feedback_terms_and_skips_excluded(self):
        terms = prf_terms(["alpha"], None, self.index, max_df_fraction=1.0)
        names = [stem for stem, _ in terms]
        self.assertEqual(names[0], "beta")
        self.assertIn("gamma", names)
        self.assertNotIn("alpha", names)
        self.assertNotIn("zeta", names)

    def test_title_zone_query(self):
        self.assertEqual(self.index.execute_boolean(("TITLE", "delta")), set(range(6, 11)))
        self.assertEqual(self.index.get_df_table(("TITLE", "delta")), {"title:delta": 5})
        self.assertEqual(self.index.estimate_hits(("TITLE", "delta")), 5)


class ParserTests(unittest.TestCase):
    def test_title_prefix_parses_to_title_node(self):
        tree = parse("covid AND title:vaccines")
        self.assertEqual(tree, ("AND", [("TERM", "covid"), ("TITLE", "vaccin")]))
        self.assertEqual(ast_to_string(tree), "(covid AND title:vaccin)")


class NoNetworkTests(unittest.TestCase):
    def test_engine_does_not_import_network_modules(self):
        for path in glob.glob(os.path.join("engine", "*.py")):
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = {alias.name for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    names = {node.module or ""}
                else:
                    continue
                self.assertFalse(names & NETWORK_MODULES, f"{path} imports {names & NETWORK_MODULES}")


if __name__ == "__main__":
    unittest.main()
