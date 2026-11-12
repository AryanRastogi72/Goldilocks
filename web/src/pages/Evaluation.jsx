import { useEffect, useState } from "react";
import * as api from "../api.js";
import { ClassBars, HitHistogram, RunBars, TopicDifferences, VersionBars } from "./charts.jsx";
import { formatFixed, formatNumber } from "../lib.js";

const FIELDS = [
  ["metadata.query", "keyword query"],
  ["text", "question text"],
];
const FIELD_NAME = Object.fromEntries(FIELDS);
const AI_TITLE = "Evidence classifier: agreement with reference labels, not human-validated accuracy";

const signed = (value, digits = 4) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
const pValue = (p) => (p < 0.0001 ? "< 0.0001" : p.toFixed(4));
const pText = (p) => (p < 0.0001 ? "p < 0.0001" : `p = ${p.toFixed(4)}`);

function findTest(significance, field, metric, a, b) {
  return significance.tests.find((t) => t.field === field && t.metric === metric && t.run_a === a && t.run_b === b);
}

function AgentVerdict({ significance }) {
  return (
    <div className="verdict" role="region" aria-label="Where the agent loses">
      <h3>Where the agent loses</h3>
      <p className="m-0">
        The EvidenceLens agent (R4) does not beat plain BM25 over the whole collection (R1). What it gives instead is a short, readable result set with every decision written down.
        Both comparisons below are paired randomization tests on nDCG@10 over the same 50 topics.
      </p>
      <ul className="mb-0 mt-2">
        {FIELDS.map(([field, name]) => {
          const t = findTest(significance, field, "ndcg_at_10", "R1", "R4");
          const clear = t.significant_after_bonferroni && t.mean_difference > 0;
          return (
            <li key={field}>
              <b>{name}:</b> BM25 {formatFixed(t.mean_a)} against agent {formatFixed(t.mean_b)} ({signed(t.mean_difference)}, {pText(t.p_value)}; BM25 wins {t.topics_a_wins} topics,
              the agent {t.topics_b_wins}, {t.topics_tied} tied).{" "}
              {clear ? "On this field the agent is clearly worse." : "On this field the two are statistically tied; the agent is not better."}
            </li>
          );
        })}
      </ul>
      <p className="mb-0 mt-2 text-sm">The Search page has a switch to rank with BM25 over the whole collection instead of the agent.</p>
    </div>
  );
}

function SummaryTable({ runs }) {
  return FIELDS.map(([field, name]) => (
    <div key={field} className="scroll-x mt-4">
      <table className="data">
        <caption>Ranking metrics on the {name}, mean over 50 topics</caption>
        <thead>
          <tr>
            <th>Run</th>
            <th>P@10</th>
            <th>nDCG@10</th>
            <th>MRR@10</th>
            <th>MAP@100</th>
            <th>Recall@100</th>
            <th>Recall@1000</th>
            <th>Latency s</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(runs).map(([run, info]) => {
            const m = info.fields[field].mean;
            return (
              <tr key={run} data-agent={run === "R4" || run === "R5"}>
                <td className="left">
                  <b>{run}</b> <span className="pencil">{info.description}</span>
                </td>
                <td className="num">{formatFixed(m.p_at_10)}</td>
                <td className="num">{formatFixed(m.ndcg_at_10)}</td>
                <td className="num">{formatFixed(m.mrr_at_10)}</td>
                <td className="num">{formatFixed(m.map_at_100)}</td>
                <td className="num">{formatFixed(m.recall_at_100)}</td>
                <td className="num">{formatFixed(m.recall_at_1000)}</td>
                <td className="num">{formatFixed(info.fields[field].mean_latency_s, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  ));
}

function SetTable({ runs }) {
  const setRuns = Object.entries(runs).filter(([, info]) => info.kind === "set");
  return (
    <div className="scroll-x mt-4">
      <table className="data">
        <caption>Set runs: the whole result set, not a ranking</caption>
        <thead>
          <tr>
            <th>Run and field</th>
            <th>Mean hits</th>
            <th>Set precision</th>
            <th>Set recall</th>
            <th>Set F1</th>
            <th>In band 20 to 200</th>
          </tr>
        </thead>
        <tbody>
          {setRuns.flatMap(([run, info]) =>
            FIELDS.map(([field, name]) => {
              const m = info.fields[field].mean;
              return (
                <tr key={`${run}${field}`}>
                  <td className="left">
                    <b>{run}</b> {name}
                  </td>
                  <td className="num">{m.hits.toFixed(1)}</td>
                  <td className="num">{formatFixed(m.set_precision)}</td>
                  <td className="num">{formatFixed(m.set_recall)}</td>
                  <td className="num">{formatFixed(m.set_f1)}</td>
                  <td className="num">{Math.round(m.in_band * 100)}%</td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
      <p className="m-0 mt-2 text-sm">
        R4 and R5 return one accepted set of about 200 papers at most, while the 50 topics average 493 judged-relevant papers. Their Recall@1000 can therefore never match a ranking run.
      </p>
    </div>
  );
}

function SignificanceTable({ significance }) {
  return (
    <div className="scroll-x mt-4">
      <table className="data">
        <caption>
          Paired randomization tests ({significance.method}). Bonferroni threshold for {significance.tests_run} tests: {significance.alpha} / {significance.tests_run} ={" "}
          {significance.bonferroni_threshold.toFixed(4)}
        </caption>
        <thead>
          <tr>
            <th>Field</th>
            <th>Metric</th>
            <th>Pair</th>
            <th>Mean difference</th>
            <th>p</th>
            <th>First wins</th>
            <th>Second wins</th>
            <th>Tied</th>
            <th>Significant after Bonferroni</th>
          </tr>
        </thead>
        <tbody>
          {significance.tests.map((t) => (
            <tr key={`${t.field}${t.metric}${t.run_a}${t.run_b}`}>
              <td className="left">{FIELD_NAME[t.field]}</td>
              <td className="left">{t.metric === "ndcg_at_10" ? "nDCG@10" : "P@10"}</td>
              <td className="left">
                {t.run_a} vs {t.run_b}
              </td>
              <td className="num">{signed(t.mean_difference)}</td>
              <td className="num">{pValue(t.p_value)}</td>
              <td className="num">{t.topics_a_wins}</td>
              <td className="num">{t.topics_b_wins}</td>
              <td className="num">{t.topics_tied}</td>
              <td className="left">{t.significant_after_bonferroni ? "yes" : "no"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Beir({ beir }) {
  return (
    <div className="sheet">
      <div className="scroll-x">
      <table className="data">
        <caption>BEIR comparison on the question text, nDCG@10</caption>
        <tbody>
          <tr>
            <td className="left">BEIR BM25 on TREC-COVID (Thakur et al. 2021, arXiv 2104.08663, {beir.paper_version_read}, Table 2)</td>
            <td className="num">{formatFixed(beir.bm25_trec_covid_ndcg10, 3)}</td>
          </tr>
          <tr>
            <td className="left">EvidenceLens R1 (BM25, full collection) on the question text</td>
            <td className="num">{formatFixed(beir.our_r1_text_ndcg_at_10, 4)}</td>
          </tr>
        </tbody>
      </table>
      </div>
      <p className="mb-0 mt-3">
        Caveat: this is close agreement, not a reproduction. BEIR's BM25 is Anserini with k1 {beir.beir_k1} and b {beir.beir_b} and separate title and text fields; ours uses k1 1.2, b 0.75
        and one token stream. We did not test which difference matters and we did not tune k1 or b on the topics. Only version v4 of the paper was read.
      </p>
    </div>
  );
}

function WinsLosses({ data }) {
  const table = (rows, caption) => (
    <div className="scroll-x mt-3">
      <table className="data">
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th>Topic</th>
            <th>R1 nDCG@10</th>
            <th>R4 nDCG@10</th>
            <th>Difference</th>
            <th className="left">R4 final Boolean query</th>
            <th>R4 hits</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.topic}>
              <td className="left">
                <b>{row.topic}</b> {row.text}
              </td>
              <td className="num">{formatFixed(row.r1_ndcg_at_10, 3)}</td>
              <td className="num">{formatFixed(row.r4_ndcg_at_10, 3)}</td>
              <td className="num">{signed(row.difference, 3)}</td>
              <td className="left mono" style={{ overflowWrap: "anywhere" }}>
                {row.final_query}
              </td>
              <td className="num">{row.hits}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
  return FIELDS.map(([field, name]) => (
    <div key={field} className="mt-4">
      <h3>On the {name}</h3>
      {table(data[field].r4_loses_to_r1_most, "The five topics where the agent loses to BM25 by the most")}
      {table(data[field].r4_beats_r1_most, "The five topics where the agent beats BM25 by the most")}
    </div>
  ));
}

function Chart({ number, children, caption }) {
  return (
    <figure className="chart" aria-label={`Chart ${number}`}>
      {children}
      <figcaption>
        <b>Chart {number}.</b> {caption}
      </figcaption>
    </figure>
  );
}

function ClassifierBlock({ block }) {
  const { human, ai } = block;
  const classRows = (perClass) => (
    <tbody>
      {Object.entries(perClass).map(([name, row]) => (
        <tr key={name}>
          <td className="left">{name.replaceAll("_", " ")}</td>
          {["precision", "recall", "f1"].map((key) => (
            <td key={key} className="num">
              {row[key] === null ? "n/a" : formatFixed(row[key], 3)}
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  );
  const header = (
    <thead>
      <tr>
        <th>Class</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1</th>
      </tr>
    </thead>
  );
  return (
    <div className="sheet">
      <h3 className="mt-0">{AI_TITLE}</h3>
      <p className="m-0">
        <b>Human-validated accuracy: {human.status ?? "TODO(metric)"}</b>. Nobody has labelled the sample by hand yet. The numbers below are not accuracy.
      </p>
      {ai.status === "not available" ? (
        <p>The AI agreement scores have not been computed.</p>
      ) : (
        <>
          <p className="mt-2">
            {ai.labelled_rows} sampled papers with abstracts were labelled blind to the classifier's guesses. Overall agreement: <b>{formatFixed(ai.weighted_accuracy, 3)}</b> re-weighted to the corpus, {formatFixed(ai.unweighted_accuracy, 3)} unweighted. Precision is unbiased
            by design of the sample; recall and agreement are estimates.
          </p>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="scroll-x">
              <table className="data">
                <caption>Re-weighted by inverse sampling probability</caption>
                {header}
                {classRows(ai.weighted_per_class)}
              </table>
            </div>
            <div className="scroll-x">
              <table className="data">
                <caption>Unweighted sample numbers</caption>
                {header}
                {classRows(ai.unweighted_per_class)}
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function ControllerVersions({ versions }) {
  return (
    <div className="scroll-x">
      <table className="data">
        <caption>Controller v2.0 and v2.1 on all 50 topics (agent run, no clarification)</caption>
        <thead>
          <tr>
            <th>Field</th>
            <th>Version</th>
            <th>In band</th>
            <th>Mean steps</th>
            <th>P@10</th>
            <th>nDCG@10</th>
          </tr>
        </thead>
        <tbody>
          {FIELDS.flatMap(([field, name]) =>
            [
              ["v2.0", versions.v2_0[field]],
              ["v2.1 (revised)", versions.v2_1[field]],
            ].map(([label, s]) => (
              <tr key={`${field}${label}`}>
                <td className="left">{name}</td>
                <td className="left">{label}</td>
                <td className="num">{Math.round(s.in_band_rate * 100)}%</td>
                <td className="num">{s.mean_steps.toFixed(2)}</td>
                <td className="num">{formatFixed(s.mean_p_at_10, 3)}</td>
                <td className="num">{formatFixed(s.mean_ndcg_at_10, 4)}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function Evaluation() {
  const [state, setState] = useState({ status: "loading" });
  useEffect(() => {
    api
      .evaluation()
      .then((data) => setState({ status: "ready", data }))
      .catch((failure) => setState({ status: "failed", message: failure.message }));
  }, []);

  if (state.status === "loading") return <p aria-live="polite">Loading the saved evaluation…</p>;
  if (state.status === "failed")
    return (
      <p className="margin-note" role="alert">
        <strong>The evaluation could not be loaded.</strong> {state.message} Start the server with <span className="mono">uvicorn api.main:app</span> and reload.
      </p>
    );
  const { data } = state;
  const charts = data.charts;
  return (
    <div className="read">
      <h1>Evaluation</h1>
      <p>
        Every number comes from saved run files for the 50 TREC-COVID topics ({formatNumber(171332)} papers), re-scored by a validator. A paper counts as relevant when its judgement is above 0.
        Nothing was tuned on these topics, and there are no held-out topics. R1 to R5 are defined in the table; R4 is the EvidenceLens agent.
      </p>
      <AgentVerdict significance={data.significance} />

      <h2>Results</h2>
      <SummaryTable runs={data.runs} />
      <SetTable runs={data.runs} />

      <h2>Do the differences count?</h2>
      <p>
        A paired randomization test asks whether a difference between two runs could come from which topics we happen to have. With {data.significance.tests_run} tests, only p below{" "}
        {data.significance.bonferroni_threshold.toFixed(4)} counts.
      </p>
      <SignificanceTable significance={data.significance} />

      <h2>Compared with the BEIR paper</h2>
      <Beir beir={data.beir} />

      <h2>Where the agent wins and loses, topic by topic</h2>
      <WinsLosses data={data.wins_and_losses} />

      <h2>Charts</h2>
      <div className="grid gap-5 lg:grid-cols-2">
        <Chart number={1} caption="nDCG@10 (solid) and P@10 (hatched) of each run, on both topic fields. R4 is the agent.">
          <RunBars data={charts["1_metrics_per_run"]} runs={data.runs} />
        </Chart>
        <Chart number={2} caption="Per-topic nDCG@10 of the agent minus BM25 on the keyword query, sorted. Below the line (hatched) BM25 is better.">
          <TopicDifferences data={charts["2_topic_differences_r4_minus_r1"]} />
        </Chart>
        <Chart number={3} caption="Final hit counts of the agent. The shaded band is the target size of 20 to 200 papers.">
          <HitHistogram data={charts["3_r4_hit_counts"]} />
        </Chart>
        <Chart number={4} caption="Papers in each study-type class, over the whole collection, by the rule-based classifier.">
          <ClassBars data={charts["4_evidence_classes"]} />
        </Chart>
      </div>

      <h2>Controller version 2.0 and 2.1</h2>
      <p>
        <b>Post-hoc revision.</b> v2.1 tightening order was revised after seeing v2.0 results on all 50 topics; there are no held-out topics. The v2.1 numbers are therefore optimistic, and the
        v2.0 numbers are kept next to them.
      </p>
      <div className="grid gap-5 lg:grid-cols-2">
        <ControllerVersions versions={data.controller_versions} />
        <Chart number={5} caption="In-band rate, nDCG@10 and P@10 for both controller versions (v2.0 hatched, v2.1 solid).">
          <VersionBars data={charts["5_controller_v2_0_vs_v2_1"]} />
        </Chart>
      </div>

      <h2>The study-type classifier</h2>
      <ClassifierBlock block={data.evidence_classifier} />
    </div>
  );
}
