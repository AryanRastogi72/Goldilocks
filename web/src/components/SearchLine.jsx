import { formatNumber, plainRule, pretty } from "../lib.js";

// A short plain clause about the turn, for the researcher. The reasoning itself stays in the sheet.
function clauses(turn) {
  return turn.resolution_rules.map(plainRule);
}

// The only line above the results: what was searched, how many papers, and the way to see how.
export default function SearchLine({ turn, onOpen, buttonRef }) {
  return (
    <div className="search-line-box">
      <div className="search-line">
        <p className="search-line-text" title={pretty(turn.rewritten_query)}>
          <span className="written-label">I searched for </span>
          <b className="written-inline">{pretty(turn.rewritten_query) || "(nothing searchable)"}</b>
          <span className="pencil"> · {formatNumber(turn.hits)} {turn.hits === 1 ? "paper" : "papers"}</span>
          {turn.ranker === "bm25" && <span className="pencil"> (BM25, whole collection)</span>}
        </p>
        <button type="button" className="btn-quiet" ref={buttonRef} onClick={onOpen}>
          How did I get this?
        </button>
      </div>
      {clauses(turn).length > 0 && <p className="rule-plain m-0 mt-1">{clauses(turn).join(" ")}</p>}
    </div>
  );
}
