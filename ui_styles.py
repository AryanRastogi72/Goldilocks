import streamlit as st


STYLE = """
<style>
:root {
    --violet: #6C5CE7;
    --teal: #14B8A6;
    --amber: #F59E0B;
    --coral: #F43F5E;
    --blue: #3B82F6;
    --slate: #1F2937;
    --muted: #596579;
    --paper: #F7F6FB;
    --white: #FFFFFF;
    --line: #E7E6EF;
}

html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: var(--paper);
    color: var(--slate);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1320px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: #F0EFF8;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 0.45rem;
}

.brand-mark {
    display: grid;
    place-items: center;
    width: 2.65rem;
    height: 2.65rem;
    border: 1px solid #D8D3FA;
    border-radius: 0.9rem;
    background: #EFEDFF;
    color: var(--violet);
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
}

.brand-name {
    color: var(--slate);
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
    font-size: 1.2rem;
    font-weight: 760;
    letter-spacing: -0.035em;
}

.brand-tagline {
    margin: 0.35rem 0 1rem 0;
    color: var(--muted);
    font-size: 0.98rem;
}

.status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0 0 1.35rem 0;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    padding: 0.42rem 0.72rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: #FFFFFF;
    color: #344054;
    font-size: 0.78rem;
    font-weight: 630;
}

.status-dot {
    width: 0.48rem;
    height: 0.48rem;
    border-radius: 50%;
    background: var(--blue);
}

.status-dot.good { background: var(--teal); }
.status-dot.warn { background: var(--amber); }
.status-dot.bad { background: var(--coral); }
.status-dot.brand { background: var(--violet); }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 1.1rem;
    border-bottom: 1px solid var(--line);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    height: 3.1rem;
    padding: 0 0.3rem;
    color: var(--muted);
    font-size: 0.94rem;
    font-weight: 640;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--violet);
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    height: 3px;
    border-radius: 3px 3px 0 0;
    background: var(--violet);
}

h1, h2, h3, h4, [data-testid="stMetricValue"] {
    color: var(--slate);
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
    letter-spacing: -0.03em;
}

[data-testid="stMetric"] {
    min-height: 6.1rem;
    padding: 1rem 1.1rem;
    border: 1px solid var(--line);
    border-radius: 1rem;
    background: var(--white);
    box-shadow: 0 5px 16px rgb(31 41 55 / 5%);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 0.81rem;
}

[data-testid="stMetricValue"] {
    font-size: 1.55rem;
    font-variant-numeric: tabular-nums;
    font-weight: 760;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    min-height: 2.9rem;
    border: 1px solid #D8D7E3;
    border-radius: 0.8rem;
    background: var(--white);
    color: var(--slate);
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--violet);
    box-shadow: 0 0 0 3px rgb(108 92 231 / 14%);
}

button[kind="primary"] {
    min-height: 2.9rem;
    border: 1px solid var(--violet);
    border-radius: 0.8rem;
    background: var(--violet);
    color: #FFFFFF;
    font-weight: 690;
    box-shadow: 0 4px 10px rgb(108 92 231 / 18%);
}

button[kind="primary"]:hover {
    border-color: #5848CF;
    background: #5848CF;
}

button[kind="secondary"],
button[kind="tertiary"] {
    border-radius: 0.75rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line);
    border-radius: 1rem;
    background: var(--white);
    box-shadow: 0 5px 18px rgb(31 41 55 / 5%);
}

[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 0.8rem;
}

[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 0.85rem;
    background: #FFFFFF;
}

[data-testid="stStatusWidget"] {
    border-radius: 0.9rem;
}

[data-testid="stAlert"] {
    border-radius: 0.85rem;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

.section-title {
    margin: 1.65rem 0 0.25rem 0;
    color: var(--slate);
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
    font-size: 1.18rem;
    font-weight: 730;
    letter-spacing: -0.025em;
}

.section-copy {
    margin: 0 0 0.9rem 0;
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.55;
}

.mono, code, pre, [data-testid="stCode"] code {
    font-family: "JetBrains Mono", Consolas, monospace !important;
}

.gauge-topline {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0.4rem 0 0.55rem 0;
    color: var(--muted);
    font-size: 0.78rem;
}

.gauge-track {
    position: relative;
    height: 0.8rem;
    border: 1px solid #DFDEE9;
    border-radius: 999px;
    background: linear-gradient(90deg, #FDE8ED 0%, #FDE8ED var(--target-start), #D9F7F2 var(--target-start), #D9F7F2 var(--target-end), #FFF1D6 var(--target-end), #FFF1D6 100%);
}

.gauge-marker {
    position: absolute;
    top: -0.38rem;
    left: var(--marker);
    width: 0.2rem;
    height: 1.5rem;
    border: 1px solid #FFFFFF;
    border-radius: 999px;
    background: var(--slate);
    box-shadow: 0 1px 5px rgb(31 41 55 / 28%);
    transition: left 320ms cubic-bezier(0.2, 0.8, 0.2, 1);
}

.gauge-zone-labels {
    display: grid;
    grid-template-columns: var(--target-start) calc(var(--target-end) - var(--target-start)) calc(100% - var(--target-end));
    margin-top: 0.45rem;
    color: var(--muted);
    font-size: 0.72rem;
    text-align: center;
}

.step-head {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
}

.step-number {
    display: grid;
    flex: 0 0 2rem;
    place-items: center;
    width: 2rem;
    height: 2rem;
    border: 1px solid #DFDDF4;
    border-radius: 0.7rem;
    background: #F1F0FC;
    color: var(--violet);
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 0.77rem;
    font-weight: 720;
}

.step-query {
    overflow-wrap: anywhere;
    color: var(--slate);
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 0.88rem;
    line-height: 1.55;
}

.decision-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    margin-top: 0.6rem;
    padding: 0.34rem 0.62rem;
    border: 1px solid transparent;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 740;
    letter-spacing: 0.015em;
}

.decision-icon {
    display: grid;
    place-items: center;
    width: 1rem;
    height: 1rem;
    border: 1px solid currentColor;
    border-radius: 50%;
    font-size: 0.68rem;
    line-height: 1;
}

.decision-accept { color: #087D70; border-color: #BCEDE5; background: #E5FAF6; }
.decision-tighten { color: #C62846; border-color: #FBCBD5; background: #FFF0F3; }
.decision-relax { color: #A56500; border-color: #F8DDA9; background: #FFF8E9; }
.decision-stop { color: #596579; border-color: #DDE1E8; background: #F3F5F8; }
.decision-info { color: #2463BD; border-color: #C9DDFB; background: #EEF5FF; }

.step-reason {
    margin: 0.7rem 0 0.75rem 0;
    color: #455166;
    font-size: 0.87rem;
    line-height: 1.55;
}

.step-counts {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.55rem;
}

.count-pill {
    padding: 0.3rem 0.52rem;
    border: 1px solid var(--line);
    border-radius: 0.55rem;
    background: #FAFAFC;
    color: #455166;
    font-size: 0.74rem;
    font-variant-numeric: tabular-nums;
}

.term-chip-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.48rem;
    margin-top: 0.75rem;
}

.term-chip {
    min-width: 7.6rem;
    padding: 0.4rem 0.52rem;
    border: 1px solid #E5E5EF;
    border-radius: 0.7rem;
    background: #FAFAFD;
}

.term-chip-top {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    color: #414B5D;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 0.7rem;
}

.df-track {
    height: 0.24rem;
    margin-top: 0.35rem;
    overflow: hidden;
    border-radius: 999px;
    background: #E8EAF0;
}

.df-fill {
    height: 100%;
    border-radius: inherit;
    background: var(--violet);
}

.result-title {
    margin: 0 0 0.35rem 0;
    color: var(--slate);
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.4;
}

.title-preview {
    margin: 0.4rem 0;
    color: #596579;
    font-size: 0.82rem;
    line-height: 1.5;
}

.title-preview mark {
    border-radius: 0.2rem;
    background: #E9E4FF;
    color: #4938B2;
    font-weight: 650;
}

.result-meta {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    margin-top: 0.4rem;
    color: var(--muted);
    font-size: 0.75rem;
}

.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 1rem;
}

.pipeline-card {
    min-height: 8.7rem;
    padding: 0.85rem;
    border: 1px solid var(--line);
    border-radius: 1rem;
    background: #FFFFFF;
    box-shadow: 0 4px 14px rgb(31 41 55 / 4%);
}

.pipeline-name {
    margin-bottom: 0.55rem;
    color: var(--violet);
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
    font-size: 0.82rem;
    font-weight: 720;
    line-height: 1.35;
}

.pipeline-concept {
    color: #596579;
    font-size: 0.75rem;
    line-height: 1.5;
}

.honesty-note {
    padding: 0.85rem 1rem;
    border: 1px solid #F5D7A5;
    border-radius: 0.85rem;
    background: #FFFAED;
    color: #76510D;
    font-size: 0.86rem;
    line-height: 1.55;
}

.eval-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.55rem;
    margin-top: 0.7rem;
}

.eval-item {
    min-width: 0;
    padding: 0.58rem 0.62rem;
    border: 1px solid #ECEBF2;
    border-radius: 0.72rem;
    background: #FBFAFD;
}

.eval-label {
    color: var(--muted);
    font-size: 0.68rem;
    line-height: 1.3;
    overflow-wrap: anywhere;
}

.eval-value {
    margin-top: 0.22rem;
    color: var(--slate);
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
    font-size: 1.02rem;
    font-weight: 750;
    font-variant-numeric: tabular-nums;
    overflow-wrap: anywhere;
}

.privacy-note {
    padding: 0.8rem;
    border: 1px solid #DDD9FA;
    border-radius: 0.8rem;
    background: #F8F7FF;
    color: #4F4A70;
    font-size: 0.78rem;
    line-height: 1.5;
}

.empty-state {
    padding: 1.35rem;
    border: 1px dashed #D8D5EA;
    border-radius: 1rem;
    background: #FBFAFF;
    color: #596579;
    text-align: center;
}

.empty-title {
    margin-bottom: 0.35rem;
    color: var(--slate);
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
    font-weight: 720;
}

.animate-step {
    animation: step-enter 360ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

@keyframes step-enter {
    from { opacity: 0; transform: translateY(7px); filter: blur(2px); }
    to { opacity: 1; transform: translateY(0); filter: blur(0); }
}

*:focus-visible {
    outline: 3px solid #4C3BC6 !important;
    outline-offset: 3px !important;
}

::selection { background: #DED9FF; color: #1F2937; }

@media (max-width: 1050px) {
    .pipeline-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}

@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem 0.9rem 2.5rem 0.9rem; }
    .pipeline-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .status-row { gap: 0.4rem; }
    .status-chip { font-size: 0.72rem; }
}

@media (max-width: 420px) {
    .pipeline-grid { grid-template-columns: 1fr; }
    .status-chip { padding: 0.34rem 0.52rem; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
</style>
"""


def inject_styles():
    st.markdown(STYLE, unsafe_allow_html=True)
