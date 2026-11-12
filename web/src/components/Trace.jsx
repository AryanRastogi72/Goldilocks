import { motion, useReducedMotion } from "motion/react";
import { LINE_FORM_WORDS, RAIL_TICKS, bandPhrase, formatNumber, lineForm, railPosition } from "../lib.js";

const EASE = [0.16, 1, 0.3, 1];
const LABELLED_TICKS = [1, 100, 10000];
const STEP_DURATION = 0.28;
const MAX_TOTAL = 0.6;

function motionFor(index, count, reduced, animate) {
  if (reduced || !animate) return { initial: false };
  const stagger = count > 1 ? Math.min(0.09, (MAX_TOTAL - 0.3) / (count - 1)) : 0;
  const delay = index * stagger;
  return {
    stagger,
    delay,
    body: {
      initial: { opacity: 0, clipPath: "inset(0 100% 0 0)" },
      animate: { opacity: 1, clipPath: "inset(0 0% 0 0)" },
      transition: { duration: STEP_DURATION, delay, ease: EASE },
    },
    bar: {
      initial: { scaleX: 0 },
      animate: { scaleX: 1 },
      transition: { duration: 0.22, delay: delay + 0.08, ease: EASE },
    },
  };
}

export function LineLegend() {
  return (
    <p className="legend" aria-label="How the line between steps is drawn">
      {["solid", "dashed", "dotted", "struck"].map((form) => (
        <span key={form}>
          <i data-form={form} />
          {LINE_FORM_WORDS[form]}
        </span>
      ))}
    </p>
  );
}

export default function Trace({ steps, band, animate = true }) {
  const reduced = useReducedMotion();
  const [low, high] = band;
  const bandStart = railPosition(low) * 100;
  const bandWidth = (railPosition(high) - railPosition(low)) * 100;

  return (
    <div className="trace" role="list" aria-label="Agent steps">
      {steps.map((step, index) => {
        const form = index === steps.length - 1 ? "end" : lineForm(step.decision);
        const hasQuery = step.query !== "";
        const out = hasQuery && (step.actual_hits < low || step.actual_hits > high);
        const m = motionFor(index, steps.length, reduced, animate);
        const wrap = (extra, children, className) =>
          m.body ? (
            <motion.div className={className} {...m.body} {...extra}>
              {children}
            </motion.div>
          ) : (
            <div className={className}>{children}</div>
          );
        return (
          <div className="trace-row" role="listitem" key={`${step.step}-${step.decision}`}>
            {wrap(
              {},
              <>
                <b aria-label={`Step ${step.step}`}>{step.step}</b>
                <span className="step-line" data-form={form} aria-hidden="true" />
              </>,
              "step-number",
            )}
            {wrap(
              {},
              <>
                <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <span>
                    <span className="decision" style={form === "struck" ? { textDecoration: "line-through", textDecorationColor: "var(--color-margin)" } : undefined}>
                      {step.decision}
                    </span>
                    <span className="source-tag" title="Who proposed this move">{step.source}</span>
                  </span>
                  {hasQuery && (
                    <span className="hits-figure">
                      {formatNumber(step.actual_hits)} {step.actual_hits === 1 ? "hit" : "hits"}{out ? `, ${bandPhrase(step.actual_hits, band)}` : ""}
                    </span>
                  )}
                </div>
                <div className="step-reason">{step.reason}</div>
                {hasQuery && (
                  <details className="step-details">
                    <summary>Boolean query of this step</summary>
                    <div className="step-query">{step.query}</div>
                  </details>
                )}
              </>,
              "step-body",
            )}
            <div className="rail-cell">
              <div className="rail-band" style={{ left: `${bandStart}%`, width: `${bandWidth}%` }} aria-hidden="true" />
              {hasQuery &&
                (m.bar ? (
                  <motion.div className="rail-bar" data-out={out} style={{ width: `${Math.max(railPosition(step.actual_hits) * 100, 1.2)}%` }} {...m.bar} />
                ) : (
                  <div className="rail-bar" data-out={out} style={{ width: `${Math.max(railPosition(step.actual_hits) * 100, 1.2)}%` }} />
                ))}
              {out && <span className="rail-flag">{bandPhrase(step.actual_hits, band)}</span>}
            </div>
          </div>
        );
      })}
      <div style={{ gridColumn: "1 / 3" }} />
      <div className="rail-axis" aria-hidden="true">
        {RAIL_TICKS.map((tick) => (
          <span key={tick} style={{ left: `${railPosition(tick) * 100}%` }}>
            {LABELLED_TICKS.includes(tick) ? (tick >= 1000 ? `${tick / 1000}k` : tick) : "\u00a0"}
          </span>
        ))}
        <span className="band-label" style={{ left: `${bandStart + bandWidth / 2}%` }}>
          band {low}-{high}
        </span>
      </div>
    </div>
  );
}
