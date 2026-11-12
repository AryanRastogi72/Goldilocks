import { forwardRef } from "react";
import { motion, useReducedMotion } from "motion/react";
import { EVIDENCE_CLASSES } from "../lib.js";
import { followLink } from "../router.js";

const MAX_QUESTION = 300;

const NAV = [
  ["/", "Search", "search"],
  ["/how", "How it works", "how"],
  ["/evaluation", "Evaluation", "evaluation"],
];

function NavLink({ href, label, active }) {
  const reduced = useReducedMotion();
  return (
    <a className="nav-link" href={href} aria-current={active ? "page" : undefined} onClick={(event) => followLink(event, href)}>
      {label}
      {active && <motion.span className="nav-underline" layoutId="nav-underline" transition={reduced ? { duration: 0 } : { duration: 0.18, ease: [0.23, 1, 0.32, 1] }} />}
    </a>
  );
}

const Head = forwardRef(function Head({ route, query, onQuery, onSubmit, loading, filter, onFilter, canReset, onReset }, inputRef) {
  const searching = route === "search";
  const tooLong = query.length > MAX_QUESTION;
  return (
    <header className="page-head">
      <div className="mx-auto flex max-w-[84rem] flex-wrap items-center gap-x-4 gap-y-1 px-4 pt-2 sm:px-6">
        <a className="brand" href="/" aria-label="EvidenceLens, home" onClick={(event) => followLink(event, "/")}>
          Evidence<span>Lens</span>
        </a>
        <nav aria-label="Pages" className="order-3 flex w-full flex-wrap sm:order-none sm:w-auto">
          {NAV.map(([href, label, id]) => (
            <NavLink key={id} href={href} label={label} active={route === id} />
          ))}
        </nav>
        {searching && canReset && (
          <button type="button" className="btn-quiet ml-auto sm:order-none" onClick={onReset}>
            Start a new page
          </button>
        )}
      </div>
      <form
          role="search"
          inert={!searching}
          aria-hidden={!searching}
          className={`mx-auto flex max-w-[84rem] flex-wrap items-center gap-2 px-4 pb-2 pt-2 sm:px-6 ${searching ? "" : "invisible"}`}
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
            aria-invalid={tooLong}
            autoComplete="off"
            placeholder="Ask, or follow up   ( / )"
            onChange={(event) => onQuery(event.target.value)}
          />
          <button type="submit" className="btn" disabled={loading || tooLong}>
            {loading ? "Searching" : "Search"}
          </button>
          <label className="flex items-center gap-2 text-sm">
            <span className="sr-only">Study type</span>
            <select className="select select-compact" aria-label="Study type" value={filter} onChange={(event) => onFilter(event.target.value)}>
              <option value="">Any study type</option>
              {EVIDENCE_CLASSES.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          {tooLong && (
            <p id="length-note" className="m-0 w-full text-sm" style={{ color: "var(--color-margin-text)" }} role="alert">
              Questions can be up to {MAX_QUESTION} characters. This one has {query.length}.
            </p>
          )}
      </form>
    </header>
  );
});

export default Head;
