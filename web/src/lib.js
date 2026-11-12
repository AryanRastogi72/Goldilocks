export const EVIDENCE_CLASSES = [
  ["systematic_review_meta_analysis", "Systematic review / meta-analysis"],
  ["randomized_controlled_trial", "Randomized controlled trial"],
  ["observational", "Observational study"],
  ["case_report_series", "Case report / series"],
  ["modelling_simulation", "Modelling / simulation"],
  ["lab_animal", "Lab / animal work"],
  ["opinion_editorial", "Opinion / editorial"],
  ["other", "Study type not detected"],
];

const STAMP_LABELS = {
  systematic_review_meta_analysis: "Systematic review",
  randomized_controlled_trial: "Randomized trial",
  observational: "Observational",
  case_report_series: "Case report / series",
  modelling_simulation: "Modelling",
  lab_animal: "Lab / animal",
  opinion_editorial: "Opinion / editorial",
  other: "Type not detected",
};

export const stampLabel = (evidenceClass) => STAMP_LABELS[evidenceClass] ?? evidenceClass;
export const evidenceLabel = (evidenceClass) => (EVIDENCE_CLASSES.find(([id]) => id === evidenceClass) ?? [0, evidenceClass])[1];

export const TOTAL_DOCUMENTS = 171332;
export const RAIL_TICKS = [1, 10, 100, 1000, 10000, 100000];

// Position of a hit count on the shared log rail, 0 to 1. The same scale is used by every step.
export const railPosition = (hits) => Math.min(1, Math.log10(Math.max(hits, 1)) / Math.log10(TOTAL_DOCUMENTS));

export const pretty = (label) => String(label).replaceAll("_", " ");

export const formatNumber = (value) => new Intl.NumberFormat("en-US").format(value);
export const formatFixed = (value, digits = 4) => Number(value).toFixed(digits);

// How the line to the next step is drawn: the form carries the meaning as well as colour.
export function lineForm(decision) {
  if (decision.startsWith("TIGHTEN")) return "dashed";
  if (decision === "RELAX_DROP_CONCEPT") return "struck";
  if (decision.startsWith("RELAX")) return "dotted";
  if (decision === "ACCEPT") return "solid";
  return "end";
}

export const LINE_FORM_WORDS = { solid: "accepted", dashed: "tightened", dotted: "relaxed", struck: "dropped", end: "stopped" };

export function bandPhrase(hits, band) {
  if (hits > band[1]) return "above band";
  if (hits < band[0]) return "below band";
  return "in band";
}

// Splits a sentence into plain and highlighted pieces from the API's character offsets.
export function highlightPieces(sentence, offsets) {
  const pieces = [];
  let cursor = 0;
  [...offsets]
    .sort((a, b) => a[0] - b[0])
    .forEach(([start, end]) => {
      if (start < cursor) return;
      if (start > cursor) pieces.push({ text: sentence.slice(cursor, start), marked: false });
      pieces.push({ text: sentence.slice(start, end), marked: true });
      cursor = end;
    });
  if (cursor < sentence.length) pieces.push({ text: sentence.slice(cursor), marked: false });
  return pieces;
}

export function longDateLabel(date = new Date()) {
  return date.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric" });
}

export const EXAMPLE_QUERIES = ["covid vaccine pregnant", "masks", "what is the origin of covid-19"];
