import { useEffect, useState } from "react";
import * as api from "../api.js";
import Trace, { LineLegend } from "../components/Trace.jsx";
import { Concepts, RulesFired } from "../components/Entry.jsx";
import { formatNumber, pretty } from "../lib.js";

const STEPS = [
  {
    name: "Conversation manager",
    plain:
      "Reads your turn. If it is a follow-up (\"what about…\", \"only…\", a pronoun) it keeps the earlier concepts and applies one of five named rules. A new topic starts a fresh page. It writes back \"I searched for…\" so you can check it understood you.",
    concept: "Query rewriting with session context, done by rules instead of a model.",
    lecture: "Not in the slides; built from named rules (follow-up cue, facet swap, exclusion, evidence filter, topic switch by Jaccard overlap).",
  },
  {
    name: "Query understanding",
    plain:
      "Drops question words, stems the rest, groups synonyms from a small curated lexicon, and makes one concept per idea. Concepts are ordered by idf, so the rarest idea comes first. A concept found in more than a tenth of the papers is held back: it would not narrow anything.",
    concept: "Tokenization, Porter stemming, stop words, idf over the union of a concept's terms.",
    lecture: "Vocabulary and stemming (02voc); tf-idf weighting and idf (lecture6-tfidf).",
  },
  {
    name: "Agent controller",
    plain:
      "Builds a Boolean query (AND of each concept's OR-group), estimates the hit count from document frequencies, runs it and counts the real hits. Inside the 20 to 200 band it accepts. Too many: it adds the next concept, requires a term in the title, or adds a feedback term. Too few: it adds thesaurus synonyms, expands a stem with the k-gram index, or drops the rarest concept. It never repeats a move, stops after 8 steps and logs each one.",
    concept: "Boolean retrieval with sorted postings, dictionary statistics, wildcard expansion, title zone, pseudo-relevance feedback, word2vec neighbours.",
    lecture: "Postings, zones and top-K (Scoring and Result Assembly); k-gram index for wildcards (03dict); Rocchio centroid (14vcat_slides).",
  },
  {
    name: "Clarifying question",
    plain:
      "When the set stays far too large, the best 300 candidates are grouped into clusters and each cluster is labelled with the words that set it apart. You pick one, and its words become a required concept.",
    concept: "Leader-follower clustering of ltc vectors, with labels from the difference between cluster and candidate means.",
    lecture: "Cluster pruning with random leaders (Scoring and Result Assembly); flat clustering (16flat_updated).",
  },
  {
    name: "Ranker",
    plain: "Orders the papers in the set with Okapi BM25 (k1 1.2, b 0.75) on all of the question's terms. Optionally it adds a small weight for the study type; by default that weight is 0.",
    concept: "BM25 term weighting with length normalisation; heap-based top-K.",
    lecture: "Top-K with a heap (Scoring and Result Assembly). BM25 itself is not in the slides we have.",
  },
  {
    name: "Evidence lens and explainer",
    plain:
      "A rule-based classifier labels each paper's study type from its title and abstract (14 ordered patterns, 8 classes). \"Why this result\" shows how much each matched term added to the score, the rule that fired, and the abstract sentence closest to your question.",
    concept: "Rule-based classification, term-contribution breakdown, sentence scoring by cosine.",
    lecture: "Evaluation of classifiers uses precision and recall (08eval). The classifier has no human-validated accuracy yet.",
  },
];

function Box({ x, y, w, h, title, note, dashed, lines = 1 }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} fill="#fbfcf9" stroke="#1d3a8a" strokeWidth="2" strokeDasharray={dashed ? "6 4" : undefined} />
      <text x={x + 12} y={y + 22} style={{ fontFamily: "var(--font-sans)", fontSize: 14, fontWeight: 700, fill: "#1d3a8a" }}>
        {title}
      </text>
      {note.map((line, index) => (
        <text key={index} x={x + 12} y={y + 40 + index * 15} style={{ fontFamily: "var(--font-sans)", fontSize: 12, fill: "#0f1a14" }}>
          {line}
        </text>
      ))}
    </g>
  );
}

function Arrow({ x, y1, y2, dashed }) {
  return <line x1={x} y1={y1} x2={x} y2={y2} stroke="#1d3a8a" strokeWidth="2" strokeDasharray={dashed ? "6 4" : undefined} markerEnd="url(#head)" />;
}

function PipelineDiagram() {
  return (
    <svg viewBox="0 0 600 680" role="img" aria-labelledby="pipeline-title pipeline-desc" className="w-full">
      <title id="pipeline-title">The EvidenceLens pipeline</title>
      <desc id="pipeline-desc">
        A question goes to the conversation manager, then query understanding, then the agent controller loop, then the ranker, the explainer and the notebook entry. The controller reads the index and may send a clarifying question back to you.
      </desc>
      <defs>
        <marker id="head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="#1d3a8a" />
        </marker>
      </defs>
      <Box x={20} y={10} w={300} h={48} title="Your turn" note={["a question and an optional study-type filter"]} />
      <Arrow x={170} y1={58} y2={82} />
      <Box x={20} y={82} w={300} h={64} title="Conversation manager" note={["follow-up rules, topic switch", "writes back \"I searched for…\""]} />
      <Arrow x={170} y1={146} y2={170} />
      <Box x={20} y={170} w={300} h={64} title="Query understanding" note={["stems, lexicon groups, concepts by idf", "common concepts held back"]} />
      <Arrow x={170} y1={234} y2={258} />
      <Box x={20} y={258} w={300} h={140} dashed title="Agent controller (at most 8 steps)" note={["build Boolean query", "estimate hits, run, count hits", "in band 20 to 200: accept", "too many: tighten", "too few: relax"]} />
      <path d="M320 318 C 352 318, 352 290, 322 290" fill="none" stroke="#1d3a8a" strokeWidth="2" strokeDasharray="2 5" markerEnd="url(#head)" />
      <text x={344} y={312} style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "#4f6055" }}>repeat</text>
      <Arrow x={170} y1={398} y2={430} />
      <Box x={20} y={430} w={300} h={48} title="Ranker" note={["BM25 inside the set"]} />
      <Arrow x={170} y1={478} y2={502} />
      <Box x={20} y={502} w={300} h={64} title="Explainer and evidence lens" note={["term contributions, study type", "best sentence with highlights"]} />
      <Arrow x={170} y1={566} y2={590} />
      <Box x={20} y={590} w={300} h={64} title="Notebook entry" note={["rewritten query, steps, result slips"]} />

      <rect x={400} y={258} width={186} height={140} fill="none" stroke="#4f6055" strokeWidth="1.5" strokeDasharray="3 4" />
      <text x={410} y={278} style={{ fontFamily: "var(--font-sans)", fontSize: 13, fontWeight: 700, fill: "#0f1a14" }}>Index, built once</text>
      {["postings and df", "title zone", "k-gram index", "word2vec thesaurus", "abstracts, evidence labels"].map((line, index) => (
        <text key={line} x={410} y={300 + index * 17} style={{ fontFamily: "var(--font-sans)", fontSize: 12, fill: "#0f1a14" }}>
          {line}
        </text>
      ))}
      <line x1={400} y1={350} x2={324} y2={350} stroke="#4f6055" strokeWidth="1.5" strokeDasharray="3 4" markerEnd="url(#head)" />

      <path d="M320 388 C 590 410, 600 90, 322 34" fill="none" stroke="#c4352b" strokeWidth="2" strokeDasharray="6 4" markerEnd="url(#head)" />
      <text x={340} y={190} style={{ fontFamily: "var(--font-sans)", fontSize: 12, fill: "#a42a21", fontWeight: 600 }}>
        still too broad:
      </text>
      <text x={340} y={206} style={{ fontFamily: "var(--font-sans)", fontSize: 12, fill: "#a42a21", fontWeight: 600 }}>
        clarifying question
      </text>
    </svg>
  );
}

function LiveExample() {
  const [state, setState] = useState({ status: "loading" });
  useEffect(() => {
    let cancelled = false;
    api
      .search({ text: "covid vaccine pregnant", session_id: `how-${Math.random().toString(36).slice(2, 10)}`, ranker: "agent", evidence_filter: null })
      .then((turn) => !cancelled && setState({ status: "ready", turn }))
      .catch((failure) => !cancelled && setState({ status: "failed", message: failure.message }));
    return () => {
      cancelled = true;
    };
  }, []);

  if (state.status === "loading") return <p aria-live="polite">Running the example against the server…</p>;
  if (state.status === "failed")
    return (
      <p className="margin-note" role="alert">
        <strong>The live example could not run.</strong> {state.message} Start the server with <span className="mono">uvicorn api.main:app</span> and reload this page.
      </p>
    );
  const { turn } = state;
  return (
    <div className="sheet">
      <div className="written-label">The question was "covid vaccine pregnant". I searched for</div>
      <div className="written">{pretty(turn.rewritten_query)}</div>
      <div className="mt-2">
        <RulesFired rules={turn.resolution_rules} />
      </div>
      <div className="mt-3">
        <Concepts concepts={turn.concepts} />
      </div>
      <h3 className="mt-4">{formatNumber(turn.hits)} {turn.hits === 1 ? "paper" : "papers"} in the set after {turn.trace.length} steps</h3>
      <LineLegend />
      <div className="mt-2">
        <Trace steps={turn.trace} band={turn.band} animate />
      </div>
      <p className="mb-0 mt-3 text-sm">
        This is a real run on the real index, not a mock-up. Read the steps top to bottom: each row says how many papers the query matched, what the agent did next, and why.
      </p>
    </div>
  );
}

export default function HowItWorks() {
  return (
    <div className="read">
      <h1>How it works</h1>
      <p>
        Every question goes through the same pipeline. Each stage is a classical information-retrieval technique, and each one leaves a mark on the Search page, so you can check it. No
        generative AI runs at query time. This course project was built by Aryan Rastogi and Raghav Garg; Raghav built the original Goldilocks search it grew from.
      </p>

      <h2>The pipeline</h2>
      <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <div className="sheet">
          <PipelineDiagram />
        </div>
        <ol className="m-0 flex list-none flex-col gap-4 p-0">
          {STEPS.map((step, index) => (
            <li key={step.name} className="pipeline-step">
              <span className="slip-rank" aria-hidden="true">
                {index + 1}
              </span>
              <div>
                <h3 className="m-0">{step.name}</h3>
                <p className="m-0 mt-1">{step.plain}</p>
                <p className="m-0 mt-1 text-sm">
                  <b>IR concept:</b> {step.concept}
                </p>
                <p className="m-0 text-sm pencil">
                  <b>Lecture term:</b> {step.lecture}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>

      <h2>A real run</h2>
      <p>Here is the first question of the demo session, fetched live from the server.</p>
      <LiveExample />
    </div>
  );
}
