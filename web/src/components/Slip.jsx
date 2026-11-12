import { useId, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { documentFor } from "../api.js";
import { evidenceLabel, formatFixed, highlightPieces, stampLabel } from "../lib.js";

const LONG_TITLE = 150;

const WINDOW = 280;

// Shows about three lines around the first highlight; the whole sentence stays one click away.
function windowAround(text, highlights) {
  if (text.length <= WINDOW) return { text, highlights, clipped: false };
  const first = highlights.length ? highlights[0][0] : 0;
  let start = Math.max(0, Math.min(first - 100, text.length - WINDOW));
  let end = Math.min(text.length, start + WINDOW);
  if (start > 0) start = text.indexOf(" ", start) + 1 || start;
  if (end < text.length) end = text.lastIndexOf(" ", end) > start ? text.lastIndexOf(" ", end) : end;
  return {
    text: text.slice(start, end),
    highlights: highlights.filter(([from, to]) => from >= start && to <= end).map(([from, to]) => [from - start, to - start]),
    clipped: true,
    before: start > 0,
    after: end < text.length,
  };
}

function Sentence({ text, highlights }) {
  const [full, setFull] = useState(false);
  const view = full ? { text, highlights, clipped: true } : windowAround(text, highlights);
  return (
    <p className="sentence">
      {view.before && <span className="pencil">… </span>}
      {highlightPieces(view.text, view.highlights).map((piece, index) =>
        piece.marked ? <mark key={index}>{piece.text}</mark> : <span key={index}>{piece.text}</span>,
      )}
      {view.after && <span className="pencil"> …</span>}
      {view.clipped && (
        <>
          {" "}
          <button type="button" className="link-button text-sm" aria-expanded={full} onClick={() => setFull(!full)}>
            {full ? "Show less" : "Show more"}
          </button>
        </>
      )}
    </p>
  );
}

function TermParts({ terms, score }) {
  const reduced = useReducedMotion();
  const largest = Math.max(...terms.map((t) => t.bm25 + t.title_boost), 1e-9);
  return (
    <div role="table" aria-label="Contribution of each matched term to the score">
      {terms.map((term) => {
        const part = term.bm25 + term.title_boost;
        return (
          <div className="term-row" role="row" key={term.stem}>
            <span role="cell">
              <span className="mono">{term.stem}</span>
              <span className="pencil"> {term.surface_forms.join(", ")}</span>
            </span>
            <span className="term-bar-wrap" role="cell">
              <motion.div
                className="term-bar"
                style={{ width: `${(part / largest) * 100}%` }}
                initial={reduced ? false : { scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                title={`BM25 ${formatFixed(term.bm25, 3)}${term.title_boost ? `, title boost ${formatFixed(term.title_boost, 3)}` : ""}`}
              />
            </span>
            <span role="cell" className="mono" style={{ textAlign: "right" }}>
              {formatFixed(part, 2)}
            </span>
          </div>
        );
      })}
      <div className="term-row" role="row" style={{ borderBottom: "none" }}>
        <span role="cell" className="pencil">
          term frequency: {terms.map((t) => `${t.stem} ${t.tf}`).join(", ")}
        </span>
        <span role="cell" />
        <span role="cell" className="mono" style={{ textAlign: "right", fontWeight: 600 }}>
          = {formatFixed(score, 2)}
        </span>
      </div>
    </div>
  );
}

export default function Slip({ result, sessionId }) {
  const panelId = useId();
  const [open, setOpen] = useState(false);
  const [expandedTitle, setExpandedTitle] = useState(false);
  const [showDuplicates, setShowDuplicates] = useState(false);
  const [abstract, setAbstract] = useState(null);
  const [abstractError, setAbstractError] = useState("");
  const longTitle = result.title.length > LONG_TITLE;

  async function toggleAbstract() {
    if (abstract !== null) {
      setAbstract(null);
      return;
    }
    try {
      const doc = await documentFor(result.ext_id, sessionId);
      setAbstract(doc.abstract || "");
      setAbstractError("");
    } catch (error) {
      setAbstractError(error.message);
    }
  }

  return (
    <article className="slip" aria-label={`Result ${result.rank}`}>
      <div className="flex items-start gap-3">
        <span className="slip-rank" aria-hidden="true">
          {result.rank}
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="slip-title" data-clamped={longTitle && !expandedTitle} title={result.title}>
            {result.title}
          </h3>
          {longTitle && (
            <button type="button" className="link-button text-sm" aria-expanded={expandedTitle} onClick={() => setExpandedTitle(!expandedTitle)}>
              {expandedTitle ? "Shorter title" : "Full title"}
            </button>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2">
            <span className="stamp" data-class={result.evidence_class} title={evidenceLabel(result.evidence_class)}>
              {stampLabel(result.evidence_class)}
            </span>
            <span className="mono pencil">
              {result.ext_id} · score {formatFixed(result.score, 2)}
            </span>
          </div>
        </div>
      </div>

      {result.best_sentence ? (
        <Sentence text={result.best_sentence} highlights={result.highlights} />
      ) : (
        <p className="no-abstract">No abstract on record for this paper. It was ranked on its title only, so there is no sentence to show.</p>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm">
        <button type="button" className="link-button" aria-expanded={open} aria-controls={panelId} onClick={() => setOpen(!open)}>
          {open ? "Hide why this result" : "Why this result"}
        </button>
        {result.duplicates.length > 0 && (
          <button type="button" className="link-button" aria-expanded={showDuplicates} onClick={() => setShowDuplicates(!showDuplicates)}>
            +{result.duplicates.length} duplicate record{result.duplicates.length > 1 ? "s" : ""}
          </button>
        )}
      </div>
      {showDuplicates && (
        <p className="mono m-0 mt-1" style={{ overflowWrap: "anywhere" }}>
          Same title, other ids: {result.duplicates.join(", ")}
        </p>
      )}

      {open && (
        <div className="why" id={panelId}>
          <div className="font-bold">BM25 contribution of each matched term</div>
          {result.matched_terms.length ? <TermParts terms={result.matched_terms} score={result.score} /> : <p>No query term matched.</p>}
          <div className="mt-3 font-bold">Study type</div>
          <p className="m-0">
            {evidenceLabel(result.evidence_class)}. Rule <span className="mono">{result.evidence_rule}</span>
            {result.evidence_rule === "NO_MATCH" ? ": no study-type cue was found in the title or abstract." : "."}
          </p>
          <div className="mt-3">
            <button type="button" className="link-button" aria-expanded={abstract !== null} onClick={toggleAbstract}>
              {abstract === null ? "Show full abstract" : "Hide abstract"}
            </button>
            {abstractError && (
              <span className="ml-2" style={{ color: "var(--color-margin-text)" }}>
                {abstractError}
              </span>
            )}
            {abstract !== null &&
              (abstract ? (
                <p className="mt-1" style={{ maxWidth: "68ch" }}>
                  {abstract}
                </p>
              ) : (
                <p className="no-abstract">No abstract on record.</p>
              ))}
          </div>
        </div>
      )}
    </article>
  );
}
