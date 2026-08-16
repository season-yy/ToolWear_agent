"""ToolWear 实验台的轻量视觉主题。"""

from __future__ import annotations

import streamlit as st


THEME_CSS = """
<style>
:root {
  --tw-ink: #17242d;
  --tw-muted: #60717d;
  --tw-paper: #f4f7f8;
  --tw-surface: #ffffff;
  --tw-border: #d6e0e4;
  --tw-steel: #315e73;
  --tw-teal: #217568;
  --tw-amber: #ad6c13;
  --tw-red: #ad4242;
  --tw-soft-blue: #e8f0f3;
  --tw-soft-teal: #e7f2ef;
  --tw-soft-amber: #fbf1df;
  --tw-soft-red: #f8e9e9;
}

.stApp {
  background: var(--tw-paper);
  color: var(--tw-ink);
  font-family: "Noto Sans SC", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
}

[data-testid="stSidebar"] {
  background: #edf2f3;
  border-right: 1px solid var(--tw-border);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.75rem;
}

.block-container {
  max-width: 1500px;
  padding-top: 1.5rem;
  padding-bottom: 3rem;
}

h1, h2, h3 {
  color: var(--tw-ink);
  letter-spacing: 0;
}

h1 {
  font-size: 2rem !important;
  line-height: 1.2 !important;
  font-weight: 720 !important;
}

h2 {
  font-size: 1.35rem !important;
  line-height: 1.3 !important;
  margin-top: 1.25rem !important;
}

h3 {
  font-size: 1.05rem !important;
  line-height: 1.35 !important;
}

code, pre, [data-testid="stCode"] {
  font-family: "Cascadia Mono", "Consolas", monospace !important;
}

.stButton > button,
.stFormSubmitButton > button,
[data-baseweb="select"] > div,
[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputContainer"],
textarea {
  border-radius: 5px !important;
}

.stButton > button {
  min-height: 2.35rem;
  border-color: #b9c8ce;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
  background: var(--tw-steel);
  border-color: var(--tw-steel);
}

.stButton > button:focus-visible,
.stFormSubmitButton > button:focus-visible {
  outline: 3px solid rgba(49, 94, 115, 0.3);
  outline-offset: 2px;
}

[data-testid="stMetric"] {
  background: transparent;
  border-left: 2px solid var(--tw-border);
  padding: 0.25rem 0 0.25rem 0.75rem;
}

[data-testid="stMetricValue"] {
  font-size: 1.45rem;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 1px solid var(--tw-border);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  height: 3rem;
  border-radius: 0;
  padding-left: 1.1rem;
  padding-right: 1.1rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--tw-border) !important;
  border-radius: 6px !important;
  background: var(--tw-surface);
}

.tw-kicker {
  color: var(--tw-steel);
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: 0.78rem;
  font-weight: 700;
  margin-bottom: 0.35rem;
}

.tw-subtitle {
  color: var(--tw-muted);
  font-size: 0.92rem;
  margin-top: -0.45rem;
  margin-bottom: 1rem;
}

.tw-idline {
  color: var(--tw-muted);
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: 0.76rem;
  overflow-wrap: anywhere;
}

.tw-rail {
  display: grid;
  grid-template-columns: repeat(7, minmax(84px, 1fr));
  gap: 2px;
  margin: 0.75rem 0 1.25rem;
  border: 1px solid var(--tw-border);
  background: var(--tw-border);
}

.tw-rail-step {
  min-height: 3.25rem;
  padding: 0.6rem 0.65rem;
  background: var(--tw-surface);
}

.tw-rail-step.done {
  background: var(--tw-soft-teal);
  color: #185c52;
}

.tw-rail-step.current {
  background: var(--tw-steel);
  color: #ffffff;
}

.tw-rail-number {
  display: block;
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: 0.68rem;
  opacity: 0.75;
}

.tw-rail-label {
  display: block;
  font-size: 0.82rem;
  font-weight: 650;
  margin-top: 0.1rem;
}

.tw-status {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  min-height: 1.65rem;
  padding: 0.2rem 0.55rem;
  border: 1px solid var(--tw-border);
  border-radius: 4px;
  background: var(--tw-surface);
  color: var(--tw-muted);
  font-size: 0.78rem;
  font-weight: 650;
}

.tw-status.ok {
  border-color: #a9cec5;
  background: var(--tw-soft-teal);
  color: #185c52;
}

.tw-status.warn {
  border-color: #e0c28f;
  background: var(--tw-soft-amber);
  color: #7b4b0d;
}

.tw-status.error {
  border-color: #dfb4b4;
  background: var(--tw-soft-red);
  color: #863535;
}

.tw-section-note {
  border-left: 3px solid var(--tw-steel);
  background: var(--tw-soft-blue);
  padding: 0.7rem 0.85rem;
  color: #294b5b;
  font-size: 0.88rem;
  margin: 0.5rem 0 1rem;
}

.tw-agent-line {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.5rem;
  align-items: center;
  border-bottom: 1px solid var(--tw-border);
  padding: 0.45rem 0;
  font-size: 0.82rem;
}

@media (max-width: 900px) {
  .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
  }
  .tw-rail {
    grid-template-columns: repeat(2, minmax(120px, 1fr));
  }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    padding-left: 0.65rem;
    padding-right: 0.65rem;
  }
}
</style>
"""


def apply_theme() -> None:
    """注入只影响实验台的 CSS 变量和布局规则。"""

    st.html(THEME_CSS)
