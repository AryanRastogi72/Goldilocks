import { EXAMPLE_QUERIES, evidenceLabel, pretty } from "../lib.js";

export function EmptyPage({ onExample }) {
  return <EmptyExplanation onExample={onExample} />;
}

function EmptyExplanation({ onExample }) {
  return (
    <section className="sheet" aria-label="A blank notebook page">
      <h2 className="m-0 text-lg font-bold">A blank page. Ask a question.</h2>
      <p className="m-0 pencil text-sm">Try one of these:</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {EXAMPLE_QUERIES.map((example) => (
          <button key={example} type="button" className="btn-quiet" onClick={() => onExample(example)}>
            {example}
          </button>
        ))}
      </div>
    </section>
  );
}

export function Loading({ text, firstSearch }) {
  return (
    <section className="sheet" aria-live="polite" aria-busy="true" aria-label="Searching">
      <div className="written-label">Searching for</div>
      <div className="written">{text}</div>
      <div className="ruled-skeleton" aria-hidden="true">
        <span style={{ width: "90%" }} />
        <span style={{ width: "70%" }} />
        <span style={{ width: "80%" }} />
      </div>
      {firstSearch && <p className="pencil m-0 text-sm">The first search after the server starts can take a moment while the index warms up.</p>}
    </section>
  );
}

export function BackendDown({ message, onRetry }) {
  return (
    <section className="margin-note" role="alert">
      <strong>The search server is not answering.</strong>
      <p className="mt-1">{message}</p>
      <p className="mt-0">
        Start it from the project folder with <span className="mono">uvicorn api.main:app</span>, then try again. Nothing you typed is lost.
      </p>
      <button type="button" className="btn-quiet" onClick={onRetry}>
        Try again
      </button>
    </section>
  );
}

export function ErrorNote({ message, onNewPage }) {
  return (
    <div className="margin-note" role="alert">
      <strong>That did not go through.</strong> {message}
      {onNewPage && (
        <p className="mb-0 mt-2">
          <button type="button" className="btn-quiet" onClick={onNewPage}>
            Start a new page
          </button>
        </p>
      )}
    </div>
  );
}

export function SessionLostNote() {
  return (
    <p className="margin-note m-0" role="status">
      <strong>This started a new page.</strong> The server no longer had your earlier turns (it restarted, or the session expired after an hour without use), so this question was read on its own.
    </p>
  );
}

export function ZeroResults({ turn, onClearFilter }) {
  const filtered = Boolean(turn.evidence_filter);
  return (
    <section className="margin-note" aria-label="No papers found">
      <strong>No papers to show.</strong>
      <p className="mb-0 mt-1">Nothing in the collection matched this question. Try different words, or open "How did I get this?" to see what was searched.</p>
      {filtered && (
        <p className="mb-0 mt-2">
          The study-type filter ({evidenceLabel(turn.evidence_filter)}) is on. A paper is kept only if the classifier labelled it that way, and none of the matches were.
        </p>
      )}
      {filtered && (
        <button type="button" className="btn-quiet mt-2" onClick={onClearFilter}>
          Clear the study-type filter
        </button>
      )}
    </section>
  );
}

export function BandNote({ turn }) {
  const last = turn.trace[turn.trace.length - 1];
  if (!last) return null;
  const below = last.decision === "ACCEPT" && last.reason.startsWith("evidence filter active");
  const exhausted = last.decision === "EXHAUSTED";
  if (!below && !exhausted) return null;
  return (
    <p className="margin-note m-0" role="status">
      <strong>{below ? "Below the band on purpose." : "The agent ran out of moves."}</strong> {pretty(last.reason)}
    </p>
  );
}
