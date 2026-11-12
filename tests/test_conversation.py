import unittest

from engine.clarify import build_clarifications
from engine.controller import build_ast, fresh_concepts, load_config, run_controller
from engine.conversation import SESSIONS, apply_clarification, handle_turn
from engine.query_understanding import understand
from engine.search import SearchIndex


def rule_names(result):
    return [rule["rule"] for rule in result["rules_fired"]]


class ConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = SearchIndex()
        cls.config = load_config()

    def start(self, session_id, first_turn="covid vaccine pregnant"):
        SESSIONS.pop(session_id, None)
        return handle_turn(session_id, first_turn, self.index, self.config, allow_clarify=False)

    def test_evidence_filter_is_stored_and_adds_no_words(self):
        first = self.start("evidence")
        result = handle_turn("evidence", "only reviews", self.index, self.config, allow_clarify=False)
        self.assertIn("EVIDENCE_FILTER", rule_names(result))
        self.assertEqual(result["evidence_filter"], "systematic_review_meta_analysis")
        self.assertEqual(result["boolean_query"], first["boolean_query"])
        self.assertTrue(result["rewritten_query"].endswith("[only: systematic_review_meta_analysis]"))

    def test_exclusion_adds_not_concept(self):
        self.start("exclusion", "coronavirus remdesivir")
        result = handle_turn("exclusion", "except hydroxychloroquine", self.index, self.config, allow_clarify=False)
        self.assertIn("EXCLUSION", rule_names(result))
        self.assertIn("AND NOT hydroxychloroquine", result["rewritten_query"])
        self.assertIn("NOT", result["boolean_query"])

    def test_cue_marks_a_followup(self):
        self.start("cue")
        result = handle_turn("cue", "what about breastfeeding", self.index, self.config, allow_clarify=False)
        self.assertEqual(rule_names(result), ["FOLLOWUP_DETECT"])
        self.assertIn("pregnancy", result["rewritten_query"])
        self.assertIn("breastfeeding", result["rewritten_query"])

    def test_pronoun_marks_a_followup(self):
        self.start("pronoun", "coronavirus remdesivir")
        result = handle_turn("pronoun", "is it safe for children", self.index, self.config, allow_clarify=False)
        self.assertEqual(result["rules_fired"][0]["detail"], "pronoun 'it'")

    def test_facet_swap_replaces_the_population(self):
        self.start("swap")
        result = handle_turn("swap", "what about children", self.index, self.config, allow_clarify=False)
        self.assertIn("FACET_SWAP", rule_names(result))
        self.assertIn("children", result["rewritten_query"])
        self.assertNotIn("pregnancy", result["rewritten_query"])
        self.assertIn("vaccine", result["rewritten_query"])

    def test_topic_switch_resets_the_concepts(self):
        self.start("switch")
        result = handle_turn("switch", "ace2 hypertension", self.index, self.config, allow_clarify=False)
        self.assertEqual(rule_names(result), ["TOPIC_SWITCH"])
        self.assertEqual(result["rewritten_query"], "ace2 AND hypertension")

    def test_excluded_concept_removes_documents(self):
        concepts = fresh_concepts(understand("coronavirus remdesivir", self.index, self.config))
        without = self.index.execute_boolean(build_ast(concepts, []))
        excluded = fresh_concepts(understand("hydroxychloroquine", self.index, self.config))[0]
        excluded["excluded"] = True
        with_exclusion = self.index.execute_boolean(build_ast(concepts + [excluded], []))
        self.assertLess(len(with_exclusion), len(without))
        self.assertTrue(with_exclusion <= without)

    def test_clarification_labels_are_filled_distinct_and_deterministic(self):
        concepts = understand("masks", self.index, self.config)
        docs = self.index.execute_boolean(build_ast(fresh_concepts(concepts), []))
        first = build_clarifications(docs, concepts, self.index, self.config)
        again = build_clarifications(docs, concepts, self.index, self.config)
        self.assertEqual(first, again)
        self.assertTrue(3 <= len(first) <= 4)
        self.assertTrue(all(choice["label_terms"] and choice["size"] >= self.config["clarify_min_size"] for choice in first))
        first_labels = [choice["label_stems"][0] for choice in first]
        self.assertEqual(len(set(first_labels)), len(first_labels))

    def test_clarification_is_offered_for_one_weak_concept_and_changes_the_query(self):
        SESSIONS.pop("clarify", None)
        offered = handle_turn("clarify", "masks", self.index, self.config, allow_clarify=True)
        self.assertEqual(offered["trace"][-1]["decision"], "CLARIFY")
        self.assertTrue(offered["clarifications"])
        chosen = apply_clarification("clarify", offered["clarifications"][0]["id"], self.index, self.config)
        self.assertNotEqual(chosen["boolean_query"], offered["boolean_query"])
        self.assertEqual(chosen["trace"][0]["decision"], "CLARIFY_CHOSEN")
        self.assertEqual(chosen["rules_fired"][0]["rule"], "CLARIFY_CHOSEN")

    def test_clarification_is_not_offered_when_not_allowed(self):
        ranked, log, _ = run_controller("masks", self.index, self.config, allow_clarify=False)
        self.assertEqual(log.clarifications, [])
        self.assertTrue(ranked)


if __name__ == "__main__":
    unittest.main()
