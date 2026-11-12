import Trace, { LineLegend } from "./Trace.jsx";
import { evidenceLabel, formatNumber, longDateLabel, pretty } from "../lib.js";

const STATUS_WORDS = {
  left_out: "left out",
  dropped: "dropped",
  excluded: "NOT",
};

const STATUS_HELP = {
  left_out: "Left out of the Boolean query (held back or too common to narrow anything).",
  dropped: "Dropped to widen the result set.",
  excluded: "Excluded: documents containing it are removed.",
};

export function Concepts({ concepts }) {
  return (
    <div>
      <ul className="m-0 flex list-none flex-wrap gap-2 p-0" aria-label="Concepts of the question">
        {concepts.map((concept) => (
          <li key={concept.label}>
            <span className="chip" data-status={concept.status} title={STATUS_HELP[concept.status] ?? "Part of the Boolean query."}>
              {pretty(concept.label)}
              {concept.required && <span className="mono">required</span>}
              {STATUS_WORDS[concept.status] && <span className="mono" style={{ textDecoration: "none" }}>{STATUS_WORDS[concept.status]}</span>}
            </span>
          </li>
        ))}
      </ul>
      <details className="mt-2 text-sm">
        <summary className="link-button inline">Stems and idf of each concept</summary>
        <ul className="m-0 mt-1 list-none p-0">
          {concepts.map((concept) => (
            <li key={concept.label} className="mono" style={{ overflowWrap: "anywhere" }}>
              <b>{pretty(concept.label)}</b> idf {concept.idf.toFixed(2)} [{concept.stems.join(" ")}] source {concept.source}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

export function RulesFired({ rules }) {
  if (!rules.length) return null;
  return (
    <ul className="m-0 flex list-none flex-col gap-1 p-0" aria-label="Follow-up rules that fired">
      {rules.map((rule) => (
        <li key={rule.rule} className="rule-note">
          <b>{rule.rule}</b>
          <span>{rule.detail}</span>
        </li>
      ))}
    </ul>
  );
}

export default function Entry({ turn, number, animate }) {
  const rewritten = turn.rewritten_query;
  return (
    <article className="entry" aria-label={`Notebook entry ${number}`}>
      <div className="entry-date">
        <span>Entry {number}</span>
        <span>{longDateLabel()}</span>
      </div>
      <div className="mt-2 pl-10">
        <div className="written-label">{turn.asked.startsWith("#") ? "You chose a clarification. I searched for" : "I searched for"}</div>
        <div className="written">{pretty(rewritten) || "(nothing searchable)"}</div>
        <div className="mt-2">
          <RulesFired rules={turn.resolution_rules} />
        </div>
      </div>
      <div className="mt-3 pl-10">
        <Concepts concepts={turn.concepts} />
      </div>
      <details className="mt-4" open>
        <summary className="flex cursor-pointer flex-wrap items-baseline justify-between gap-2 pl-10">
          <h2 className="m-0 inline text-base font-bold">
            Steps{turn.ranker === "bm25" ? "" : `: ${formatNumber(turn.hits)} ${turn.hits === 1 ? "paper" : "papers"} in the set`}
          </h2>
          <span className="mono pencil">{Math.round(turn.timings.total_ms)} ms</span>
        </summary>
        <div className="mt-1 pl-10">
          <LineLegend />
        </div>
        <div className="mt-2">
          <Trace steps={turn.trace} band={turn.band} animate={animate} />
        </div>
      </details>
      <div className="mt-3 pl-10 text-sm">
        {turn.ranker === "bm25" ? (
          <p className="m-0">
            <b>BM25 over the whole collection.</b> No Boolean set was built; {formatNumber(turn.hits)} papers contain a concept stem.
          </p>
        ) : (
          <>
            <div className="pencil">Boolean query that was run</div>
            <div className="mono" style={{ overflowWrap: "anywhere" }}>{turn.boolean_query || "(none)"}</div>
          </>
        )}
        {turn.evidence_filter && (
          <p className="m-0 mt-2">
            Only <b>{evidenceLabel(turn.evidence_filter)}</b> papers are kept.
          </p>
        )}
      </div>
    </article>
  );
}

export function Stub({ turn, number }) {
  return (
    <details className="stub" style={{ display: "block" }}>
      <summary className="cursor-pointer list-none" style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", gap: "4px 10px", alignItems: "baseline" }}>
        <span className="mono pencil">Entry {number}</span>
        <span className="written">{pretty(turn.rewritten_query) || turn.asked}</span>
        <span className="mono pencil">{turn.ranker === "bm25" ? "BM25" : `${formatNumber(turn.hits)} papers`}</span>
      </summary>
      <div className="mt-2 text-sm">
        <RulesFired rules={turn.resolution_rules} />
        <div className="mt-2">
          <Trace steps={turn.trace} band={turn.band} animate={false} />
        </div>
      </div>
    </details>
  );
}
