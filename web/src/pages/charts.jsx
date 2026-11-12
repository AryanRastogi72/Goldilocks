import { formatNumber, stampLabel } from "../lib.js";

// Hand-drawn SVG charts from the JSON of /api/eval. Two series are told apart by fill (solid or hatched), not by colour alone.

const INK = "#1d3a8a";
const PAPER = "#0f1a14";
const PENCIL = "#4f6055";
const MARGIN = "#c4352b";
const GRID = "rgb(79 96 85 / 0.35)";
const FIELD_LABELS = { "metadata.query": "keyword query", text: "question text" };

function Defs() {
  return (
    <defs>
      <pattern id="hatch-ink" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="#fbfcf9" />
        <rect width="2.4" height="6" fill={INK} />
      </pattern>
      <pattern id="hatch-pencil" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="#fbfcf9" />
        <rect width="2.4" height="6" fill={PENCIL} />
      </pattern>
      <pattern id="hatch-margin" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="#fbfcf9" />
        <rect width="2.4" height="6" fill={MARGIN} />
      </pattern>
    </defs>
  );
}

function Legend({ items, x = 40, y = 12 }) {
  return (
    <g>
      {items.map((item, index) => (
        <g key={item.label} transform={`translate(${x + index * 150}, ${y})`}>
          <rect width="16" height="10" fill={item.fill} stroke={item.stroke ?? INK} strokeWidth="1" />
          <text x="22" y="9">
            {item.label}
          </text>
        </g>
      ))}
    </g>
  );
}

function ValueAxis({ left, top, height, width, max, step, format = (v) => v.toFixed(1) }) {
  const ticks = [];
  for (let value = 0; value <= max + 1e-9; value += step) ticks.push(value);
  return (
    <g>
      {ticks.map((value) => {
        const y = top + height - (value / max) * height;
        return (
          <g key={value}>
            <line x1={left} x2={left + width} y1={y} y2={y} stroke={GRID} strokeWidth="1" />
            <text x={left - 6} y={y + 4} textAnchor="end">
              {format(value)}
            </text>
          </g>
        );
      })}
      <line x1={left} x2={left} y1={top} y2={top + height} stroke={PAPER} strokeWidth="1.5" />
    </g>
  );
}

export function RunBars({ data, runs }) {
  const fields = Object.keys(data);
  const left = 44;
  const width = 500;
  const panelHeight = 150;
  const slot = width / data[fields[0]].runs.length;
  return (
    <svg viewBox="0 0 560 470" role="img" aria-label={`nDCG at 10 and precision at 10 for runs R0 to R5, ${fields.map((f) => FIELD_LABELS[f]).join(" and ")}`}>
      <Defs />
      <Legend items={[{ label: "nDCG@10", fill: INK }, { label: "P@10", fill: "url(#hatch-ink)", stroke: INK }]} />
      {fields.map((field, panel) => {
        const top = 64 + panel * 196;
        const entry = data[field];
        return (
          <g key={field}>
            <text x={left} y={top - 14} style={{ fontWeight: 600 }}>
              {FIELD_LABELS[field]}
            </text>
            <ValueAxis left={left} top={top + 6} height={panelHeight} width={width} max={1} step={0.2} />
            {entry.runs.map((run, index) => {
              const x = left + index * slot;
              const barWidth = slot * 0.36;
              const bars = [
                { value: entry.ndcg_at_10[index], fill: INK, dx: slot * 0.1 },
                { value: entry.p_at_10[index], fill: "url(#hatch-ink)", dx: slot * 0.1 + barWidth + 2 },
              ];
              return (
                <g key={run}>
                  {bars.map((bar) => {
                    const h = bar.value * panelHeight;
                    return (
                      <g key={bar.dx}>
                        <rect x={x + bar.dx} y={top + 6 + panelHeight - h} width={barWidth} height={h} fill={bar.fill} stroke={INK} strokeWidth="1" />
                        <text x={x + bar.dx + barWidth / 2} y={top + 2 + panelHeight - h} textAnchor="middle" style={{ fontSize: 11 }}>
                          {bar.value.toFixed(2)}
                        </text>
                      </g>
                    );
                  })}
                  <text x={x + slot / 2} y={top + panelHeight + 22} textAnchor="middle" style={{ fontWeight: run === "R4" ? 700 : 400 }}>
                    {run}
                    {run === "R4" ? " agent" : ""}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
      <title>{Object.entries(runs).map(([name, run]) => `${name}: ${run.description}`).join("; ")}</title>
    </svg>
  );
}

export function TopicDifferences({ data }) {
  const left = 46;
  const width = 500;
  const top = 26;
  const height = 190;
  const max = Math.max(1, ...data.differences.map(Math.abs));
  const mid = top + height / 2;
  const slot = width / data.differences.length;
  const y = (value) => (value / max) * (height / 2);
  return (
    <svg viewBox="0 0 560 270" role="img" aria-label="Per-topic difference in nDCG at 10 between the agent and BM25, keyword query, sorted">
      <Defs />
      <text x={left + 4} y={top - 8}>agent better</text>
      <text x={left + 4} y={top + height + 14}>BM25 better</text>
      {[-1, -0.5, 0, 0.5, 1].map((tick) => (
        <g key={tick}>
          <line x1={left} x2={left + width} y1={mid - y(tick)} y2={mid - y(tick)} stroke={tick === 0 ? PAPER : GRID} strokeWidth={tick === 0 ? 1.5 : 1} />
          <text x={left - 6} y={mid - y(tick) + 4} textAnchor="end">
            {tick > 0 ? `+${tick}` : tick}
          </text>
        </g>
      ))}
      {data.differences.map((value, index) => (
        <rect
          key={data.topics[index]}
          x={left + index * slot + 1}
          y={value >= 0 ? mid - y(value) : mid}
          width={slot - 2}
          height={Math.max(Math.abs(y(value)), 0.5)}
          fill={value >= 0 ? INK : "url(#hatch-margin)"}
          stroke={value >= 0 ? INK : MARGIN}
          strokeWidth="1"
        >
          <title>{`Topic ${data.topics[index]}: ${value >= 0 ? "+" : ""}${value.toFixed(3)}`}</title>
        </rect>
      ))}
      <text x={left + width / 2} y={262} textAnchor="middle">
        50 topics, sorted by difference
      </text>
    </svg>
  );
}

const HIT_EDGES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000];

export function HitHistogram({ data }) {
  const fields = Object.keys(data).filter((key) => key !== "band");
  const [low, high] = data.band;
  const left = 40;
  const width = 500;
  const panelHeight = 96;
  const xFor = (value) => left + (Math.log10(value) / 5) * width;
  return (
    <svg viewBox="0 0 560 290" role="img" aria-label="How many topics ended with each final hit count for the agent. The band from 20 to 200 is shaded.">
      <Defs />
      {fields.map((field, panel) => {
        const top = 20 + panel * 118;
        const counts = HIT_EDGES.slice(0, -1).map((edge, i) => data[field].hits.filter((h) => h >= edge && h < HIT_EDGES[i + 1]).length);
        const max = Math.max(10, ...counts);
        return (
          <g key={field}>
            <text x={left} y={top - 6} style={{ fontWeight: 600 }}>
              {FIELD_LABELS[field]}
            </text>
            <rect x={xFor(low)} y={top} width={xFor(high) - xFor(low)} height={panelHeight} fill="rgb(29 58 138 / 0.13)" stroke={INK} strokeWidth="1.5" />
            <line x1={left} x2={left} y1={top} y2={top + panelHeight} stroke={PAPER} strokeWidth="1.5" />
            <line x1={left} x2={left + width} y1={top + panelHeight} y2={top + panelHeight} stroke={PAPER} strokeWidth="1.5" />
            {counts.map((count, i) => (
              <g key={HIT_EDGES[i]}>
                <rect x={xFor(HIT_EDGES[i]) + 1} y={top + panelHeight - (count / max) * panelHeight} width={xFor(HIT_EDGES[i + 1]) - xFor(HIT_EDGES[i]) - 2} height={(count / max) * panelHeight} fill={INK} />
                {count > 0 && (
                  <text x={(xFor(HIT_EDGES[i]) + xFor(HIT_EDGES[i + 1])) / 2} y={top + panelHeight - (count / max) * panelHeight - 3} textAnchor="middle" style={{ fontSize: 11 }}>
                    {count}
                  </text>
                )}
              </g>
            ))}
          </g>
        );
      })}
      {[1, 10, 100, 1000, 10000, 100000].map((tick) => (
        <text key={tick} x={xFor(tick)} y={262} textAnchor="middle">
          {tick >= 1000 ? `${tick / 1000}k` : tick}
        </text>
      ))}
      <text x={left + width / 2} y={282} textAnchor="middle">
        final hit count (log scale), shaded = band {low} to {high}
      </text>
    </svg>
  );
}

export function ClassBars({ data }) {
  const left = 168;
  const width = 300;
  const rowHeight = 26;
  const max = Math.max(...data.documents);
  return (
    <svg viewBox={`0 0 560 ${data.classes.length * rowHeight + 40}`} role="img" aria-label="Number of papers in each evidence class">
      {data.classes.map((name, index) => {
        const value = data.documents[index];
        const w = (value / max) * width;
        return (
          <g key={name} transform={`translate(0, ${10 + index * rowHeight})`}>
            <text x={left - 8} y={15} textAnchor="end">
              {stampLabel(name)}
            </text>
            <rect x={left} y={3} width={w} height={17} fill={INK} />
            <text x={left + w + 6} y={16}>
              {formatNumber(value)}
            </text>
          </g>
        );
      })}
      <line x1={left} x2={left} y1={6} y2={data.classes.length * rowHeight + 8} stroke={PAPER} strokeWidth="1.5" />
      <text x={left} y={data.classes.length * rowHeight + 30}>
        papers, linear scale, total {formatNumber(data.total)}
      </text>
    </svg>
  );
}

export function VersionBars({ data }) {
  const fields = Object.keys(data);
  const left = 44;
  const width = 500;
  const panelHeight = 120;
  const slot = width / data[fields[0]].metrics.length;
  return (
    <svg viewBox="0 0 560 390" role="img" aria-label="Controller version 2.0 compared with version 2.1: in-band rate, nDCG at 10 and P at 10, both fields">
      <Defs />
      <Legend items={[{ label: "v2.0", fill: "url(#hatch-pencil)", stroke: PENCIL }, { label: "v2.1 (revised)", fill: INK }]} />
      {fields.map((field, panel) => {
        const top = 66 + panel * 168;
        const entry = data[field];
        return (
          <g key={field}>
            <text x={left} y={top - 14} style={{ fontWeight: 600 }}>
              {FIELD_LABELS[field]}
            </text>
            <ValueAxis left={left} top={top + 6} height={panelHeight} width={width} max={1} step={0.5} />
            {entry.metrics.map((metric, index) => {
              const x = left + index * slot;
              const barWidth = slot * 0.3;
              return (
                <g key={metric}>
                  {[entry.v2_0[index], entry.v2_1[index]].map((value, series) => (
                    <g key={series}>
                      <rect
                        x={x + slot * 0.15 + series * (barWidth + 3)}
                        y={top + 6 + panelHeight - value * panelHeight}
                        width={barWidth}
                        height={value * panelHeight}
                        fill={series === 0 ? "url(#hatch-pencil)" : INK}
                        stroke={series === 0 ? PENCIL : INK}
                      />
                      <text x={x + slot * 0.15 + series * (barWidth + 3) + barWidth / 2} y={top + 2 + panelHeight - value * panelHeight} textAnchor="middle" style={{ fontSize: 11 }}>
                        {value.toFixed(2)}
                      </text>
                    </g>
                  ))}
                  <text x={x + slot / 2} y={top + panelHeight + 22} textAnchor="middle">
                    {metric}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
