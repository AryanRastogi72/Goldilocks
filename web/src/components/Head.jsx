import { forwardRef } from "react";
import { EVIDENCE_CLASSES } from "../lib.js";

const MAX_QUESTION = 300;

const NAV = [
  ["#/", "Search", "search"],
  ["#/how", "How it works", "how"],
  ["#/evaluation", "Evaluation", "evaluation"],
];

const Head = forwardRef(function Head(
  { route, query, onQuery, onSubmit, loading, ranker, onRanker, filter, onFilter, canReset, onReset },
  inputRef,
) {
  const searching = route === "search";
  return (
    <header className="page-head">
      <div className="mx-auto flex max-w-[84rem] flex-wrap items-center gap-x-4 gap-y-1 px-4 pt-2 sm:px-6">
        <a className="brand" href="#/" aria-label="EvidenceLens, home">
          Evidence<span>Lens</span>
        </a>
        <nav aria-label="Pages" className="order-3 flex w-full flex-wrap sm:order-none sm:w-auto">
          {NAV.map(([href, label, id]) => (
            <a key={id} className="nav-link" href={href} aria-current={route === id ? "page" : undefined}>
              {label}
            </a>
          ))}
        </nav>
        {searching && canReset && (
          <button type="button" className="btn-quiet ml-auto sm:order-none" onClick={onReset}>
            Start a new page
          </button>
        )}
      </div>
      {searching && (
        <form
          role="search"
          className="mx-auto flex max-w-[84rem] flex-wrap items-center gap-2 px-4 pb-2 pt-2 sm:px-6"
          onSubmit={(event) => {
            event.preventDefault();
            onSubmit();
          }}
        >
          <label htmlFor="question" className="sr-only">
            Your question
          </label>
          <input
            id="question"
            ref={inputRef}
            className="search-field"
            type="search"
            value={query}
            aria-describedby="length-note"
            aria-invalid={query.length > MAX_QUESTION}
            autoComplete="off"
            placeholder="Ask, or follow up   ( / )"
            onChange={(event) => onQuery(event.target.value)}
          />
          <button type="submit" className="btn" disabled={loading || query.length > MAX_QUESTION}>
            {loading ? "Searching" : "Search"}
          </button>
          <fieldset className="segmented" aria-label="Ranker">
            <label>
              <input type="radio" name="ranker" value="agent" checked={ranker === "agent"} onChange={() => onRanker("agent")} />
              Agent
            </label>
            <label>
              <input type="radio" name="ranker" value="bm25" checked={ranker === "bm25"} onChange={() => onRanker("bm25")} />
              BM25, whole collection
            </label>
          </fieldset>
          {query.length > MAX_QUESTION && (
            <p id="length-note" className="m-0 w-full text-sm" style={{ color: "var(--color-margin-text)" }} role="alert">
              Questions can be up to {MAX_QUESTION} characters. This one has {query.length}.
            </p>
          )}
          <label className="flex items-center gap-2 text-sm">
            <span className="pencil">Study type</span>
            <select className="select" value={filter} onChange={(event) => onFilter(event.target.value)}>
              <option value="">Any study type</option>
              {EVIDENCE_CLASSES.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </form>
      )}
    </header>
  );
});

export default Head;
