import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from api.main import app, create_app, group_duplicates
from engine.conversation import SESSIONS, SessionStore, handle_turn

client = None


def setUpModule():
    global client
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def search(text, session_id=None, **extra):
    body = dict(text=text, **extra)
    if session_id:
        body["session_id"] = session_id
    return client.post("/api/search", json=body)


def assert_error(test, response, status, code):
    test.assertEqual(response.status_code, status)
    body = response.json()
    test.assertEqual(set(body), {"error"})
    test.assertEqual(set(body["error"]), {"code", "message"})
    test.assertEqual(body["error"]["code"], code)
    test.assertNotIn("Traceback", body["error"]["message"])


class SearchTests(unittest.TestCase):
    def test_health(self):
        body = client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["docs"], 171332)
        self.assertTrue(body["index_loaded"])

    def test_search_returns_the_documented_shape(self):
        response = search("covid vaccine pregnant")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in ["session_id", "rewritten_query", "resolution_rules", "boolean_query", "concepts", "trace", "clarifications",
                    "results", "hits", "in_band", "evidence_filter", "ranker", "timings"]:
            self.assertIn(key, body)
        self.assertEqual(body["ranker"], "agent")
        self.assertTrue(body["in_band"])
        self.assertEqual(set(body["timings"]), {"total_ms", "controller_ms", "explain_ms"})
        first = body["results"][0]
        for key in ["rank", "doc_id", "ext_id", "title", "score", "evidence_class", "evidence_rule", "matched_terms",
                    "best_sentence", "highlights", "duplicates"]:
            self.assertIn(key, first)
        self.assertEqual([r["rank"] for r in body["results"]], list(range(1, len(body["results"]) + 1)))

    def test_explanation_contributions_match_the_score(self):
        body = search("covid vaccine pregnant").json()
        for result in body["results"]:
            total = sum(t["bm25"] + t["title_boost"] for t in result["matched_terms"])
            self.assertAlmostEqual(total, result["score"], places=6)

    def test_concepts_carry_a_status_and_the_band_is_returned(self):
        body = search("covid vaccine pregnant").json()
        self.assertEqual(body["band"], [20, 200])
        statuses = {c["label"]: c["status"] for c in body["concepts"]}
        self.assertEqual(set(statuses.values()) - {"active", "left_out", "dropped", "excluded"}, set())
        self.assertEqual(statuses["pregnancy"], "active")

    def test_dropped_and_excluded_concepts_are_marked(self):
        search("coronavirus remdesivir", "api_status")
        body = search("except hydroxychloroquine", "api_status").json()
        self.assertEqual({c["label"]: c["status"] for c in body["concepts"]}["hydroxychloroquine"], "excluded")

    def test_follow_up_turn_fires_rules(self):
        first = search("covid vaccine pregnant", "api_followup").json()
        second = search("what about breastfeeding", "api_followup").json()
        self.assertEqual(second["session_id"], first["session_id"])
        self.assertEqual(second["resolution_rules"][0]["rule"], "FOLLOWUP_DETECT")
        self.assertIn("breastfeeding", second["rewritten_query"])

    def test_unknown_session_id_is_reported_as_a_new_session(self):
        SESSIONS.pop("api_never_seen", None)
        first = search("masks", "api_never_seen").json()
        again = search("masks", "api_never_seen").json()
        self.assertTrue(first["new_session"])
        self.assertFalse(again["new_session"])
        self.assertFalse(search("masks").json()["new_session"])

    def test_missing_session_id_creates_one(self):
        self.assertRegex(search("masks").json()["session_id"], r"^[A-Za-z0-9_-]{1,64}$")

    def test_masks_offers_clarifications_and_clarify_works(self):
        offered = search("masks", "api_masks").json()
        self.assertGreaterEqual(len(offered["clarifications"]), 3)
        chosen = client.post("/api/clarify", json={"session_id": "api_masks", "choice_id": offered["clarifications"][0]["id"]})
        self.assertEqual(chosen.status_code, 200)
        body = chosen.json()
        self.assertEqual(body["resolution_rules"][0]["rule"], "CLARIFY_CHOSEN")
        self.assertNotEqual(body["boolean_query"], offered["boolean_query"])

    def test_clarify_errors(self):
        assert_error(self, client.post("/api/clarify", json={"session_id": "no_such_session", "choice_id": 1}), 404, "unknown_session")
        search("covid vaccine pregnant", "api_noclar")
        assert_error(self, client.post("/api/clarify", json={"session_id": "api_noclar", "choice_id": 99}), 422, "unknown_choice")

    def test_bm25_ranker_ranks_the_whole_collection_and_keeps_the_filter(self):
        body = search("masks", ranker="bm25", evidence_filter="randomized_controlled_trial").json()
        self.assertEqual(body["ranker"], "bm25")
        self.assertEqual(body["boolean_query"], "")
        self.assertEqual([step["decision"] for step in body["trace"]], ["BM25_FULL_COLLECTION"])
        self.assertTrue(body["results"])
        self.assertTrue(all(r["evidence_class"] == "randomized_controlled_trial" for r in body["results"]))

    def test_evidence_filter_null_clears_the_session_filter(self):
        search("masks", "api_filter", evidence_filter="randomized_controlled_trial")
        cleared = search("masks", "api_filter", evidence_filter=None).json()
        self.assertIsNone(cleared["evidence_filter"])


class RerankTests(unittest.TestCase):
    def test_rerank_switches_the_ranker_in_place(self):
        agent = search("covid vaccine pregnant", "api_rerank").json()
        bm25 = client.post("/api/rerank", json={"session_id": "api_rerank", "ranker": "bm25"})
        self.assertEqual(bm25.status_code, 200)
        body = bm25.json()
        self.assertEqual(body["ranker"], "bm25")
        self.assertEqual(body["rewritten_query"], agent["rewritten_query"])
        self.assertEqual(body["trace"][0]["decision"], "BM25_FULL_COLLECTION")
        self.assertEqual(len(SESSIONS["api_rerank"]["turns"]), 1)
        back = client.post("/api/rerank", json={"session_id": "api_rerank", "ranker": "agent"}).json()
        self.assertEqual(back["boolean_query"], agent["boolean_query"])
        self.assertEqual([r["ext_id"] for r in back["results"]], [r["ext_id"] for r in agent["results"]])
        self.assertEqual(len(SESSIONS["api_rerank"]["turns"]), 1)

    def test_bm25_switch_matches_run_r1_for_a_single_question(self):
        question = "what are the benefits and risks of re-opening schools in the midst of the COVID-19 pandemic?"
        run_file = Path(__file__).resolve().parent.parent / "evaluation" / "runs" / "R1_text.trec"
        expected = [line.split()[2] for line in run_file.read_text().splitlines() if line.split()[0] == "48"][:10]
        state = client.app.state
        result = handle_turn("api_r1", question, state.index, state.config, ranker="bm25")
        self.assertEqual([r["ext_id"] for r in result["results"]], expected)

    def test_bm25_switch_uses_the_typed_words_of_the_whole_topic(self):
        search("covid vaccine pregnant", "api_topic_words")
        search("what about breastfeeding", "api_topic_words", ranker="bm25")
        self.assertTrue({"breastfeed", "pregnant", "vaccin", "covid"} <= set(SESSIONS["api_topic_words"]["ranking_stems"]))
        search("ace2 hypertension", "api_topic_words", ranker="bm25")
        self.assertNotIn("breastfeed", SESSIONS["api_topic_words"]["ranking_stems"])

    def test_rerank_errors(self):
        assert_error(self, client.post("/api/rerank", json={"session_id": "api_no_such", "ranker": "bm25"}), 404, "unknown_session")
        assert_error(self, client.post("/api/rerank", json={"session_id": "api_rerank", "ranker": "magic"}), 422, "invalid_request")
        SESSIONS.setdefault("api_empty", {"turns": [], "concepts": [], "clarifications": [], "evidence_filter": None})
        assert_error(self, client.post("/api/rerank", json={"session_id": "api_empty", "ranker": "bm25"}), 422, "no_turn")


class WarmUpTests(unittest.TestCase):
    def test_warm_up_runs_one_search_and_leaves_no_session(self):
        from api.main import warm_up
        warm_up(app.state)
        self.assertNotIn("_warm_up", SESSIONS)


class DocumentAndEvalTests(unittest.TestCase):
    def test_doc_with_an_abstract(self):
        body = client.get("/api/doc/lhbk4902").json()
        self.assertGreater(len(body["abstract"]), 100)
        self.assertIn("evidence", body["explanation"])
        self.assertFalse(body["session_used"])

    def test_doc_uses_the_session_terms(self):
        search("masks decontamination", "api_doc_session")
        body = client.get("/api/doc/lhbk4902", params={"session_id": "api_doc_session"}).json()
        self.assertTrue(body["session_used"])
        self.assertTrue(body["explanation"]["contributions"])

    def test_doc_without_an_abstract(self):
        body = client.get("/api/doc/6iu1dtyl").json()
        self.assertEqual(body["abstract"], "")
        self.assertIsNone(body["explanation"]["best_sentence"])
        self.assertEqual(body["explanation"]["best_sentence_missing_reason"], "no abstract")

    def test_unknown_document_is_404(self):
        assert_error(self, client.get("/api/doc/nonexistent"), 404, "unknown_document")

    def test_eval_carries_the_hand_labelled_classifier_scores(self):
        body = client.get("/api/eval").json()
        self.assertEqual(set(body["evidence_classifier"]), {"human"})
        human = body["evidence_classifier"]["human"]
        self.assertEqual(human["labelled_rows"], 120)
        self.assertAlmostEqual(human["weighted_accuracy"], 0.597, places=3)
        for key in ["runs", "significance", "beir", "wins_and_losses", "charts", "controller_versions"]:
            self.assertIn(key, body)
        self.assertEqual(len(body["charts"]), 5)


class ValidationTests(unittest.TestCase):
    def test_each_invalid_input_gives_the_error_shape(self):
        cases = [
            {"text": ""}, {"text": "   "}, {"text": "x" * 301}, {"text": "masks", "session_id": "bad id!"},
            {"text": "masks", "session_id": "a" * 65}, {"text": "masks", "evidence_filter": "best_studies"},
            {"text": "masks", "ranker": "magic"}, {},
        ]
        for body in cases:
            assert_error(self, client.post("/api/search", json=body), 422, "invalid_request")

    def test_text_is_trimmed_before_the_length_check(self):
        self.assertEqual(search("  masks  ").status_code, 200)
        self.assertEqual(search("x" * 300).status_code, 200)

    def test_bad_ids_in_the_path_and_query(self):
        assert_error(self, client.get("/api/doc/bad id!"), 422, "invalid_request")
        assert_error(self, client.get("/api/doc/lhbk4902", params={"session_id": "bad id!"}), 422, "invalid_request")

    def test_unknown_route_has_the_error_shape(self):
        assert_error(self, client.get("/api/nothing"), 404, "not_found")

    def test_server_errors_hide_the_stack_trace(self):
        with mock.patch("api.main.handle_turn", side_effect=RuntimeError("secret detail")):
            response = search("masks")
        assert_error(self, response, 500, "internal_error")
        self.assertNotIn("secret detail", response.text)

    def test_cors_only_for_the_vite_dev_server(self):
        allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        other = client.get("/api/health", headers={"Origin": "http://evil.example"})
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "http://localhost:5173")
        self.assertNotIn("access-control-allow-origin", other.headers)


class DuplicateTests(unittest.TestCase):
    def test_identical_normalised_titles_are_grouped(self):
        items = [
            {"ext_id": "a", "title": "Considerations for Obstetric Care!"},
            {"ext_id": "b", "title": "A different paper"},
            {"ext_id": "c", "title": "considerations for obstetric care"},
            {"ext_id": "d", "title": "Considerations for Obstetric Care"},
        ]
        groups = group_duplicates(items)
        self.assertEqual([g["ext_id"] for g in groups], ["a", "b"])
        self.assertEqual(groups[0]["duplicates"], ["c", "d"])
        self.assertEqual(groups[1]["duplicates"], [])

    def test_empty_titles_are_not_grouped(self):
        groups = group_duplicates([{"ext_id": "a", "title": "..."}, {"ext_id": "b", "title": "--"}])
        self.assertEqual(len(groups), 2)

    def test_search_ranks_count_groups(self):
        body = search("covid vaccine pregnant").json()
        self.assertTrue(any(r["duplicates"] for r in body["results"]))
        self.assertEqual([r["rank"] for r in body["results"]], list(range(1, len(body["results"]) + 1)))


class SessionStoreTests(unittest.TestCase):
    def test_idle_sessions_expire(self):
        now = [0.0]
        store = SessionStore(max_sessions=10, idle_seconds=60, clock=lambda: now[0])
        store.setdefault("a", {"n": 1})
        now[0] = 50
        store["a"]
        now[0] = 100
        self.assertIn("a", store)
        now[0] = 161
        self.assertNotIn("a", store)
        with self.assertRaises(KeyError):
            store["a"]

    def test_oldest_session_is_evicted_first(self):
        now = [0.0]
        store = SessionStore(max_sessions=2, idle_seconds=1000, clock=lambda: now[0])
        store.setdefault("a", {})
        now[0] = 1
        store.setdefault("b", {})
        now[0] = 2
        store["a"]
        now[0] = 3
        store.setdefault("c", {})
        self.assertEqual((len(store), "a" in store, "b" in store, "c" in store), (2, True, False, True))

    def test_api_evicts_with_a_small_limit(self):
        original = (SESSIONS.max_sessions, SESSIONS.idle_seconds)
        SESSIONS.configure(2, 3600)
        try:
            for name in ["evict_one", "evict_two", "evict_three"]:
                search("masks", name)
            assert_error(self, client.post("/api/clarify", json={"session_id": "evict_one", "choice_id": 1}), 404, "unknown_session")
        finally:
            SESSIONS.configure(*original)


class StaticFileTests(unittest.TestCase):
    def test_root_without_a_build_points_to_health(self):
        plain = TestClient(create_app(dist_dir=None, load_engine=False))
        self.assertEqual(plain.get("/").json()["health"], "/api/health")

    def test_built_frontend_is_served_with_a_single_page_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            dist = Path(folder) / "dist"
            (dist / "assets").mkdir(parents=True)
            (dist / "index.html").write_text("<html>app</html>", encoding="utf-8")
            (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
            (Path(folder) / "secret.txt").write_text("secret", encoding="utf-8")
            static = TestClient(create_app(dist_dir=dist, load_engine=False))
            self.assertEqual(static.get("/").text, "<html>app</html>")
            self.assertEqual(static.get("/assets/app.js").text, "console.log(1)")
            self.assertEqual(static.get("/some/client/route").text, "<html>app</html>")
            self.assertNotIn("secret", static.get("/..%2Fsecret.txt").text)
            assert_error(self, static.get("/api/unknown"), 404, "not_found")


if __name__ == "__main__":
    unittest.main()
