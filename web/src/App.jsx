import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api.js";
import Head from "./components/Head.jsx";
import Hero from "./components/Hero.jsx";
import Entry, { Stub } from "./components/Entry.jsx";
import Slip from "./components/Slip.jsx";
import Clarify from "./components/Clarify.jsx";
import { BackendDown, BandNote, EmptyPage, ErrorNote, Loading, ZeroResults } from "./components/Notes.jsx";
import HowItWorks from "./pages/HowItWorks.jsx";
import Evaluation from "./pages/Evaluation.jsx";

const ROUTES = { "#/how": "how", "#/evaluation": "evaluation" };

function useRoute() {
  const read = () => ROUTES[window.location.hash] ?? "search";
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => {
      setRoute(read());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

function isTypingTarget(element) {
  return element && (["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName) || element.isContentEditable);
}

export default function App() {
  const route = useRoute();
  const inputRef = useRef(null);
  const [query, setQuery] = useState("");
  const [ranker, setRanker] = useState("agent");
  const [filter, setFilter] = useState("");
  const [turns, setTurns] = useState([]);
  const [loadingText, setLoadingText] = useState("");
  const [error, setError] = useState("");
  const [down, setDown] = useState("");
  const [showAnyway, setShowAnyway] = useState(false);
  const sessionId = useRef(null);

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
      if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey && !isTypingTarget(document.activeElement)) {
        event.preventDefault();
        if (window.location.hash && window.location.hash !== "#/") window.location.hash = "#/";
        setTimeout(() => inputRef.current && inputRef.current.focus(), 0);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const finishTurn = useCallback((response, asked) => {
    sessionId.current = response.session_id;
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
      setLoadingText(trimmed);
      try {
        const payload = { text: trimmed, ranker, evidence_filter: evidenceFilter || null };
        if (sessionId.current) payload.session_id = sessionId.current;
        finishTurn(await api.search(payload), trimmed);
      } catch (failure) {
        fail(failure);
      } finally {
        setLoadingText("");
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
      }
    },
    [finishTurn, fail],
  );

  const reset = useCallback(() => {
    sessionId.current = null;
    setTurns([]);
    setFilter("");
    setError("");
    setShowAnyway(false);
    inputRef.current && inputRef.current.focus();
  }, []);

  const clearFilter = useCallback(() => {
    const lastQuestion = [...turns].reverse().find((turn) => !turn.asked.startsWith("#"));
    setFilter("");
    if (lastQuestion) submit(lastQuestion.asked, "");
  }, [turns, submit]);

  const current = turns[turns.length - 1];
  const earlier = turns.slice(0, -1);
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
        ranker={ranker}
        onRanker={setRanker}
        filter={filter}
        onFilter={setFilter}
        canReset={turns.length > 0}
        onReset={reset}
      />
      {route === "search" && <Hero />}
      <main id="content" className="mx-auto max-w-[84rem] px-4 pb-16 pt-3 sm:px-6">
        {route === "search" && (
          <div className="grid gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            {down && <BackendDown message={down} onRetry={checkServer} />}
            {!down && !turns.length && !loadingText && <EmptyPage onExample={(example) => submit(example)} />}
            {!down && loadingText && !turns.length && <Loading text={loadingText} />}
            {turns.length > 0 && (
              <>
                <div className="flex min-w-0 flex-col gap-3">
                  {earlier.map((turn, index) => (
                    <Stub key={index} turn={turn} number={index + 1} />
                  ))}
                  <Entry key={turns.length} turn={current} number={turns.length} animate />
                </div>
                <section className="flex min-w-0 flex-col gap-4" aria-label="Results" aria-busy={Boolean(loadingText)}>
                  {error && <ErrorNote message={error} />}
                  {loadingText && <Loading text={loadingText} />}
                  <BandNote turn={current} />
                  {asking ? (
                    <Clarify turn={current} busy={Boolean(loadingText)} onChoose={choose} onSkip={() => setShowAnyway(true)} />
                  ) : current.results.length ? (
                    <ol className="m-0 flex list-none flex-col gap-5 p-0">
                      {current.results.map((result) => (
                        <li key={`${turns.length}-${result.ext_id}`}>
                          <Slip result={result} sessionId={current.session_id} />
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <ZeroResults turn={current} onClearFilter={clearFilter} />
                  )}
                </section>
              </>
            )}
            {error && !turns.length && <ErrorNote message={error} />}
          </div>
        )}
        {route === "how" && <HowItWorks />}
        {route === "evaluation" && <Evaluation />}
      </main>
    </>
  );
}
