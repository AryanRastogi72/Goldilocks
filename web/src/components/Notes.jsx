import { EXAMPLE_QUERIES, evidenceLabel, longDateLabel } from "../lib.js";

export function EmptyPage({ onExample }) {
  return (
    <>
      <div className="entry blank-entry" aria-hidden="true">
        <div className="entry-date">
          <span>Entry 1</span>
          <span>{longDateLabel()}</span>
        </div>
        <div className="mt-2 pl-10">
          <div className="written-label">I searched for</div>
          <div className="ruled-line" />
        </div>
        <div className="mt-4 pl-10">
          <div className="written-label">Steps</div>
          <div className="ruled-line" />
          <div className="ruled-line" />
          <div className="ruled-line" />
        </div>
      </div>
      <EmptyExplanation onExample={onExample} />
    </>
  );
}

function EmptyExplanation({ onExample }) {
  return (
    <section className="sheet" aria-label="A blank notebook page">
      <h2 className="m-0 text-lg font-bold">A blank page. Ask a question.</h2>
      <p className="mt-1" style={{ maxWidth: "62ch" }}>
        Each question becomes a dated entry: what I searched for, the numbered steps the agent took to bring the result set to a readable size, and the papers it
        found, with the reason each one is there. Ask a follow-up and the new entry stacks under the old one.
      </p>
      <p className="m-0 pencil text-sm">Try one of these:</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {EXAMPLE_QUERIES.map((example) => (
          <button key={example} type="button" className="btn-quiet" onClick={() => onExample(example)}>
            {example}
          </button>
        ))}
      </div>
      <p className="mb-0 mt-3 text-sm" style={{ maxWidth: "62ch" }}>
        After the first one, try <span className="mono">what about breastfeeding</span>, then <span className="mono">only reviews</span>. Put words in double quotes to
        make them required terms.
      </p>
    </section>
  );
}

export function Loading({ text }) {
  return (
    <section className="sheet" style={{ gridColumn: "1 / -1" }} aria-live="polite" aria-busy="true" aria-label="Searching">
      <div className="written-label">Searching for</div>
      <div className="written">{text}</div>
      <div className="ruled-skeleton" aria-hidden="true">
        <span style={{ width: "90%" }} />
        <span style={{ width: "70%" }} />
        <span style={{ width: "80%" }} />
      </div>
      <p className="pencil m-0 text-sm">The first search after the server starts can take a moment while the index warms up.</p>
    </section>
  );
}

export function BackendDown({ message, onRetry }) {
  return (
    <section className="margin-note" style={{ gridColumn: "1 / -1" }} role="alert">
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

export function ErrorNote({ message }) {
  return (
    <p className="margin-note m-0" role="alert">
      <strong>That did not go through.</strong> {message}
    </p>
  );
}

export function ZeroResults({ turn, onClearFilter }) {
  const last = turn.trace[turn.trace.length - 1];
  const filtered = Boolean(turn.evidence_filter);
  return (
    <section className="margin-note" aria-label="No papers found">
      <strong>No papers to show.</strong>
      <p className="mb-0 mt-1">{last ? last.reason : "The query matched nothing."}</p>
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
      <strong>{below ? "Below the band on purpose." : "The agent ran out of moves."}</strong> {last.reason}
    </p>
  );
}
