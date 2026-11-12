import { formatNumber } from "../lib.js";

export default function Clarify({ turn, onChoose, onSkip, busy }) {
  return (
    <section aria-label="Clarifying question" className="flex flex-col gap-3">
      <div className="margin-note">
        <strong>That is too broad: {formatNumber(turn.hits)} papers match.</strong>
        <p className="mb-0 mt-1">I grouped the best 300 into clusters and labelled each with the words that set it apart. Which do you mean?</p>
      </div>
      <ul className="m-0 flex list-none flex-col gap-2 p-0">
        {turn.clarifications.map((choice) => (
          <li key={choice.id}>
            <button type="button" className="clarify-choice" disabled={busy} onClick={() => onChoose(choice)}>
              <b>{choice.label_terms.join(" / ")}</b>
              <span className="pencil text-sm">{choice.size} of the 300 papers I grouped</span>
            </button>
          </li>
        ))}
      </ul>
      <button type="button" className="link-button self-start text-sm" onClick={onSkip}>
        Skip the question and show the top {turn.results.length} anyway
      </button>
    </section>
  );
}
