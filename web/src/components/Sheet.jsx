import { useEffect, useRef } from "react";
import Entry, { Stub } from "./Entry.jsx";
import { BandNote } from "./Notes.jsx";

// The notebook entry as a side sheet (a full-screen sheet on phones). It sits over the page and never moves the results.
export default function Sheet({ turns, ranker, onRanker, busy, error, listening, onClose }) {
  const closeButton = useRef(null);
  const current = turns[turns.length - 1];
  const earlier = turns.slice(0, -1);

  useEffect(() => {
    closeButton.current && closeButton.current.focus();
  }, []);

  useEffect(() => {
    if (!listening) return undefined;
    function onKey(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [listening, onClose]);

  return (
    <>
      <aside className="sheet-panel" aria-labelledby="sheet-title">
        <div className="flex items-center justify-between gap-3">
          <h2 id="sheet-title" className="m-0 text-lg font-bold">
            How did I get this?
          </h2>
          <button type="button" className="btn-quiet" ref={closeButton} onClick={onClose}>
            Close
          </button>
        </div>

        <div className="ranker-switch">
          <label className="switch">
            <input type="checkbox" role="switch" checked={ranker === "bm25"} disabled={busy} onChange={(event) => onRanker(event.target.checked ? "bm25" : "agent")} />
            <span className="switch-track" aria-hidden="true" />
            <span>Rank the whole collection with BM25 instead</span>
          </label>
          <p className="m-0 mt-1 text-sm">
            The agent keeps the set small enough to read and may miss relevant papers on long questions. BM25 over the whole collection ranks every paper that contains one of your words, with no size limit.
            {busy ? " Ranking again…" : ""}
          </p>
          {error && (
            <p className="m-0 mt-2 text-sm" role="alert" style={{ color: "var(--color-margin-text)" }}>
              {error}
            </p>
          )}
        </div>

        <div className="mt-3">
          <BandNote turn={current} />
        </div>

        <div className="mt-4">
          <Entry key={`${turns.length}-${current.ranker}`} turn={current} number={turns.length} animate />
        </div>

        {earlier.length > 0 && (
          <section className="mt-5" aria-label="Earlier in this session">
            <h3 className="m-0 mb-2 text-base font-bold">Earlier in this session</h3>
            <div className="flex flex-col gap-2">
              {earlier.map((turn, index) => (
                <Stub key={index} turn={turn} number={index + 1} />
              ))}
            </div>
          </section>
        )}
      </aside>
    </>
  );
}
