import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import * as api from "./api.js";
import Head from "./components/Head.jsx";
import Hero from "./components/Hero.jsx";
import SearchLine from "./components/SearchLine.jsx";
import Sheet from "./components/Sheet.jsx";
import Slip from "./components/Slip.jsx";
import Clarify from "./components/Clarify.jsx";
import { BackendDown, EmptyPage, ErrorNote, Loading, SessionLostNote, ZeroResults } from "./components/Notes.jsx";
import HowItWorks from "./pages/HowItWorks.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import { navigate, useRoute } from "./router.js";

function isTypingTarget(element) {
  return element && (["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName) || element.isContentEditable);
}

// Every page stays mounted once visited and is only hidden, so its state survives a visit to another page.
function useKeptPages(route) {
  const [visited, setVisited] = useState({ [route]: true });
  useEffect(() => {
    setVisited((previous) => (previous[route] ? previous : { ...previous, [route]: true }));
  }, [route]);
  return visited;
}

function useScrollMemory(route) {
  const memory = useRef({});
  const current = useRef(route);
  useEffect(() => {
    const remember = () => {
      memory.current[current.current] = window.scrollY;
    };
    window.addEventListener("scroll", remember, { passive: true });
    return () => window.removeEventListener("scroll", remember);
  }, []);
  useLayoutEffect(() => {
    current.current = route;
    window.scrollTo(0, memory.current[route] ?? 0);
  }, [route]);
}

function useEvaluationData(route) {
  const [state, setState] = useState({ status: "idle" });
  useEffect(() => {
    if (route !== "evaluation" || state.status !== "idle") return;
    setState({ status: "loading" });
    api
      .evaluation()
      .then((data) => setState({ status: "ready", data }))
      .catch((failure) => setState({ status: "failed", message: failure.message }));
  }, [route, state.status]);
  return state;
}

export default function App() {
  const route = useRoute();
  const visited = useKeptPages(route);
  useScrollMemory(route);
  const evaluationState = useEvaluationData(route);
  const inputRef = useRef(null);
  const sheetButton = useRef(null);
  const sessionId = useRef(null);
  const [query, setQuery] = useState("");
  const [ranker, setRanker] = useState("agent");
  const [filter, setFilter] = useState("");
  const [turns, setTurns] = useState([]);
  const [loadingText, setLoadingText] = useState("");
  const [error, setError] = useState("");
  const [down, setDown] = useState("");
  const [showAnyway, setShowAnyway] = useState(false);
  const [searchedOnce, setSearchedOnce] = useState(false);
  const [sessionLost, setSessionLost] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [reranking, setReranking] = useState(false);
  const [sheetError, setSheetError] = useState("");
  const sheetOpenRef = useRef(false);
  sheetOpenRef.current = sheetOpen;

  const checkServer = useCallback(async () => {
    try {
      await api.health();
      setDown("");
    } catch (failure) {
      setDown(failure.message);
    }
  }, []);

  useEffect(() => {
    checkServer();
  }, [checkServer]);

  useEffect(() => {
    function onKey(event) {
      if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey && !isTypingTarget(document.activeElement) && !sheetOpenRef.current) {
        event.preventDefault();
        navigate("/");
        setTimeout(() => inputRef.current && inputRef.current.focus(), 0);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const finishTurn = useCallback((response, asked) => {
    sessionId.current = response.session_id;
    setSessionLost(Boolean(response.new_session));
    setTurns((previous) => [...previous, { ...response, asked }]);
    setFilter(response.evidence_filter ?? "");
    setShowAnyway(false);
    setQuery("");
    setDown("");
  }, []);

  const fail = useCallback((failure) => {
    if (failure.code === "network") setDown(failure.message);
    else setError(failure.message);
  }, []);

  const submit = useCallback(
    async (text = query, evidenceFilter = filter) => {
      const trimmed = text.trim();
      if (!trimmed) {
        inputRef.current && inputRef.current.focus();
        return;
      }
      setError("");
      setSessionLost(false);
      setLoadingText(trimmed);
      try {
        const payload = { text: trimmed, ranker, evidence_filter: evidenceFilter || null };
        if (sessionId.current) payload.session_id = sessionId.current;
        finishTurn(await api.search(payload), trimmed);
      } catch (failure) {
        fail(failure);
      } finally {
        setLoadingText("");
        setSearchedOnce(true);
      }
    },
    [query, filter, ranker, finishTurn, fail],
  );

  const choose = useCallback(
    async (choice) => {
      setError("");
      setLoadingText(choice.label_terms.join(" / "));
      try {
        finishTurn(await api.clarify({ session_id: sessionId.current, choice_id: choice.id }), `#${choice.id}`);
      } catch (failure) {
        fail(failure);
      } finally {
        setLoadingText("");
        setSearchedOnce(true);
      }
    },
    [finishTurn, fail],
  );

  // Ranks the current turn again with the other ranker. The turn is replaced in place: no new entry.
  const switchRanker = useCallback(
    async (next) => {
      const previous = ranker;
      setRanker(next);
      if (!sessionId.current || !turns.length) return;
      setReranking(true);
      setSheetError("");
      try {
        const response = await api.rerank({ session_id: sessionId.current, ranker: next });
        setTurns((all) => [...all.slice(0, -1), { ...response, asked: all[all.length - 1].asked }]);
        setShowAnyway(false);
      } catch (failure) {
        setRanker(previous);
        setSheetError(failure.message);
      } finally {
        setReranking(false);
      }
    },
    [ranker, turns.length],
  );

  const closeSheet = useCallback(() => {
    setSheetOpen(false);
    setTimeout(() => sheetButton.current && sheetButton.current.focus(), 0);
  }, []);

  const reset = useCallback(() => {
    sessionId.current = null;
    setTurns([]);
    setFilter("");
    setError("");
    setShowAnyway(false);
    setSheetOpen(false);
    setSheetError("");
    inputRef.current && inputRef.current.focus();
  }, []);

  const clearFilter = useCallback(() => {
    const lastQuestion = [...turns].reverse().find((turn) => !turn.asked.startsWith("#"));
    setFilter("");
    if (lastQuestion) submit(lastQuestion.asked, "");
  }, [turns, submit]);

  const errorIsSessionLoss = error.startsWith("This session does not exist");
  const current = turns[turns.length - 1];
  const asking = current && current.clarifications.length > 0 && !showAnyway;

  return (
    <>
      <a className="skip-link" href="#content">
        Skip to the content
      </a>
      <Head
        ref={inputRef}
        route={route}
        query={query}
        onQuery={setQuery}
        onSubmit={() => submit()}
        loading={Boolean(loadingText)}
        filter={filter}
        onFilter={setFilter}
        canReset={turns.length > 0}
        onReset={reset}
      />
      <div hidden={route !== "search"}>
        <h1 className="sr-only">Search COVID-19 papers</h1>
        {!turns.length && <Hero />}
        <main id="content" className="mx-auto max-w-[84rem] px-4 pb-16 pt-3 sm:px-6">
          <div className="results-column">
            {down && <BackendDown message={down} onRetry={checkServer} />}
            {!down && !turns.length && !loadingText && <EmptyPage onExample={(example) => submit(example)} />}
            {!down && loadingText && !turns.length && <Loading text={loadingText} firstSearch={!searchedOnce} />}
            {turns.length > 0 && (
              <>
                <SearchLine turn={current} onOpen={() => setSheetOpen(true)} buttonRef={sheetButton} />
                {error && <ErrorNote message={error} onNewPage={errorIsSessionLoss ? reset : undefined} />}
                {sessionLost && !loadingText && <SessionLostNote />}
                {loadingText && <Loading text={loadingText} firstSearch={!searchedOnce} />}
                {asking ? (
                  <Clarify turn={current} busy={Boolean(loadingText)} onChoose={choose} onSkip={() => setShowAnyway(true)} />
                ) : current.results.length ? (
                  <ol className="m-0 flex list-none flex-col gap-5 p-0">
                    {current.results.map((result) => (
                      <li key={result.ext_id}>
                        <Slip result={result} sessionId={current.session_id} />
                      </li>
                    ))}
                  </ol>
                ) : (
                  <ZeroResults turn={current} onClearFilter={clearFilter} />
                )}
              </>
            )}
            {error && !turns.length && <ErrorNote message={error} />}
          </div>
        </main>
        {sheetOpen && turns.length > 0 && (
          <Sheet turns={turns} ranker={ranker} onRanker={switchRanker} busy={reranking} error={sheetError} listening={route === "search"} onClose={closeSheet} />
        )}
      </div>
      {(visited.how || route === "how") && (
        <div hidden={route !== "how"}>
          <main className="mx-auto max-w-[84rem] px-4 pb-16 pt-3 sm:px-6">
            <HowItWorks />
          </main>
        </div>
      )}
      {(visited.evaluation || route === "evaluation") && (
        <div hidden={route !== "evaluation"}>
          <main className="mx-auto max-w-[84rem] px-4 pb-16 pt-3 sm:px-6">
            <Evaluation state={evaluationState} />
          </main>
        </div>
      )}
    </>
  );
}
