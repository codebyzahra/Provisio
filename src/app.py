"""
app.py
======
Provisio — Developer Onboarding Copilot.

Multi-step SPA implemented with st.session_state routing.

Steps:
    landing   → Welcome / hero
    input     → Repository source selection (URL or ZIP upload)
    analyzing → Progressive pipeline execution
    dashboard → Rendered onboarding guide (tabs)

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import streamlit as st

# Ensure src/ is importable regardless of CWD.
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import load_coach, load_mapper, load_mentor  # noqa: E402

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Provisio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ── Fonts ───────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Keyframe animations ─────────────────────────────────────────────────── */
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes glow-pulse {
    0%, 100% { opacity: 0.55; }
    50%       { opacity: 0.85; }
}
@keyframes gradientMove {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
@keyframes cta-glow-pulse {
    0%, 100% { box-shadow: 0 0 22px rgba(124, 58, 237, 0.55), 0 0 48px rgba(124, 58, 237, 0.2); }
    50%       { box-shadow: 0 0 38px rgba(124, 58, 237, 0.85), 0 0 72px rgba(96, 165, 250, 0.3); }
}

/* ── Base / deep dark background with radial glow ────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #09090b !important;
    color: #e4e4e7 !important;
    font-family: "Inter", system-ui, -apple-system, sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* Radial purple-blue glow at the top centre of the viewport */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: -180px;
    left: 50%;
    transform: translateX(-50%);
    width: 900px;
    height: 560px;
    border-radius: 50%;
    background: radial-gradient(
        ellipse at center,
        rgba(124, 58, 237, 0.22) 0%,
        rgba(59, 130, 246, 0.12) 45%,
        transparent 72%
    );
    pointer-events: none;
    animation: glow-pulse 5s ease-in-out infinite;
    z-index: 0;
}

[data-testid="stMain"],
[data-testid="block-container"] {
    background: transparent !important;
    position: relative;
    z-index: 1;
}

/* ── Strip ALL Streamlit chrome ──────────────────────────────────────────── */
#MainMenu,
footer,
header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stHeader"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── Hide sidebar entirely (SPA — no sidebar nav) ───────────────────────── */
[data-testid="stSidebar"] {
    display: none !important;
}

/* ── Main block container ────────────────────────────────────────────────── */
[data-testid="block-container"] {
    padding: 2.5rem 2.5rem 5rem !important;
    max-width: 1100px !important;
}

/* ── Landing-specific overrides ──────────────────────────────────────────── */
body[data-landing="true"] [data-testid="stAppViewContainer"],
.landing-bg {
    background: linear-gradient(135deg, #050505 0%, #1a0b2e 40%, #0d1b2a 70%, #050505 100%) !important;
    background-size: 200% 200% !important;
    animation: gradientMove 15s ease infinite !important;
}

/* Full-viewport flex stage */
.landing-stage {
    min-height: 85vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    text-align: center;
    padding: 2rem 1.5rem;
    animation: fade-in-up 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* Eyebrow badge */
.landing-eyebrow {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #a78bfa;
    border: 1px solid rgba(167, 139, 250, 0.35);
    border-radius: 20px;
    padding: 5px 16px;
    background: rgba(167, 139, 250, 0.08);
}

/* Massive gradient title */
.landing-title {
    font-size: 4.5rem;
    font-weight: 900;
    letter-spacing: -0.05em;
    line-height: 1.05;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 10%, #c4b5fd 45%, #60a5fa 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    max-width: 820px;
}

/* Subtitle */
.landing-subtitle {
    font-size: 1.25rem;
    color: #a1a1aa;
    line-height: 1.7;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
}

/* CTA button wrapper — keeps the st.button output centred */
.landing-cta-wrap {
    display: flex;
    justify-content: center;
    width: 100%;
}

/* Override primary button on landing page to be larger + pulsing glow */
.landing-cta-wrap [data-testid="stBaseButton-primary"] {
    font-size: 15px !important;
    padding: 0.8rem 2.4rem !important;
    border-radius: 10px !important;
    animation: cta-glow-pulse 2.8s ease-in-out infinite !important;
}
.landing-cta-wrap [data-testid="stBaseButton-primary"]:hover {
    animation: none !important;
    box-shadow: 0 0 52px rgba(124, 58, 237, 0.95), 0 0 90px rgba(96, 165, 250, 0.4) !important;
    transform: translateY(-3px) !important;
    opacity: 1 !important;
}

/* ── Fade-in-up wrapper applied via inline HTML ──────────────────────────── */
.fade-in-up {
    animation: fade-in-up 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* ── Hero / centred layout helper ───────────────────────────────────────── */
.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 5.5rem 1rem 3rem;
    max-width: 700px;
    margin: 0 auto;
    animation: fade-in-up 0.65s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.hero-eyebrow {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #a78bfa;
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 20px;
    padding: 4px 14px;
    margin-bottom: 1.75rem;
    background: rgba(167, 139, 250, 0.07);
}

/* Gradient title */
.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1.15;
    margin: 0 0 1.25rem;
    background: linear-gradient(135deg, #f4f4f5 20%, #a78bfa 60%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 1.08rem;
    color: #71717a;
    line-height: 1.75;
    margin: 0 0 2.75rem;
    font-weight: 400;
    max-width: 560px;
}

/* ── Glassmorphism card ──────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.75rem 1.6rem;
    animation: fade-in-up 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.glass-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e4e4e7;
    margin: 0 0 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.glass-card-sub {
    font-size: 12px;
    color: #52525b;
    margin: 0 0 1.25rem;
}

/* Input view legacy aliases → glass-card */
.option-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.75rem 1.5rem;
    animation: fade-in-up 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.option-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e4e4e7;
    margin: 0 0 0.35rem;
}
.option-card-sub {
    font-size: 12px;
    color: #52525b;
    margin: 0 0 1.25rem;
}

/* ── Dashboard header card ───────────────────────────────────────────────── */
.dash-header {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.4rem 1.6rem 1.2rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    animation: fade-in-up 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.dash-header-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #f4f4f5, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.2rem;
}
.dash-header-sub {
    font-size: 12px;
    color: #52525b;
    margin: 0;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    word-break: break-all;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
input[type="text"],
input[type="url"],
textarea {
    background-color: rgba(255, 255, 255, 0.04) !important;
    color: #e4e4e7 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    font-family: inherit !important;
    font-size: 13px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
input:focus,
textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.18) !important;
    outline: none !important;
}
textarea {
    font-family: "JetBrains Mono", "Fira Code", monospace !important;
    line-height: 1.5 !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    background: rgba(255, 255, 255, 0.02) !important;
    padding: 0.5rem !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(124, 58, 237, 0.5) !important;
}

/* ── Buttons — glowing gradient primary ──────────────────────────────────── */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 0 16px rgba(124, 58, 237, 0.35) !important;
    transition: opacity 0.15s ease, transform 0.12s ease, box-shadow 0.15s ease !important;
}
[data-testid="stBaseButton-primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 0 28px rgba(124, 58, 237, 0.55) !important;
}
[data-testid="stBaseButton-primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 0 10px rgba(124, 58, 237, 0.3) !important;
}
[data-testid="stBaseButton-primary"]:disabled {
    background: rgba(255,255,255,0.05) !important;
    color: #3f3f46 !important;
    opacity: 1 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Secondary / ghost button ────────────────────────────────────────────── */
[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, 0.03) !important;
    color: #71717a !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    color: #e4e4e7 !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
    background: rgba(255, 255, 255, 0.06) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07) !important;
    gap: 0;
    padding: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #52525b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.2rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #a1a1aa !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #e4e4e7 !important;
    border-bottom: 2px solid #7c3aed !important;
}
[data-testid="stTabsContent"] {
    background: transparent !important;
    padding-top: 1.75rem !important;
    animation: fade-in-up 0.4s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] p {
    color: #52525b !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #e4e4e7 !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}

/* ── Dataframes ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: rgba(255,255,255,0.02) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    font-size: 13px !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #e4e4e7 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

/* ── Status widget ───────────────────────────────────────────────────────── */
[data-testid="stStatusContainer"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 10px !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

/* ── Captions ────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    color: #52525b !important;
    font-size: 12px !important;
}

/* ── Headings ────────────────────────────────────────────────────────────── */
h1 {
    color: #e4e4e7 !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.2rem !important;
}
h2 {
    color: #d4d4d8 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin: 1.75rem 0 0.75rem !important;
}
h3 {
    color: #52525b !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    margin: 1.5rem 0 0.6rem !important;
}

/* ── Spinner container ───────────────────────────────────────────────────── */
.analyzing-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 4rem 1rem 2rem;
    max-width: 560px;
    margin: 0 auto;
    animation: fade-in-up 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.analyzing-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e4e4e7;
    margin: 1.5rem 0 0.4rem;
}
.analyzing-sub {
    font-size: 13px;
    color: #52525b;
    margin: 0 0 2rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    """Seed session_state keys on first run."""
    if "step" not in st.session_state:
        st.session_state.step = "landing"
    if "target" not in st.session_state:
        st.session_state.target = ""
    if "pipeline_error" not in st.session_state:
        st.session_state.pipeline_error = ""


def _goto(step: str) -> None:
    """Transition to *step* and rerun immediately."""
    st.session_state.step = step
    st.rerun()


def _reset() -> None:
    """Return to the landing view and clear all transient state."""
    st.session_state.step = "landing"
    st.session_state.target = ""
    st.session_state.pipeline_error = ""
    st.rerun()


# ---------------------------------------------------------------------------
# ZIP extraction utility
# ---------------------------------------------------------------------------

def _extract_zip(uploaded_file: object) -> str:
    """Extract a ZIP upload to a temp directory and return the path.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Absolute path to the extracted directory root.

    Raises:
        zipfile.BadZipFile: If the uploaded bytes are not a valid ZIP.
    """
    raw_bytes = uploaded_file.getvalue()
    zip_buffer = io.BytesIO(raw_bytes)

    with zipfile.ZipFile(zip_buffer, "r") as zf:
        extract_dir = tempfile.mkdtemp(prefix="provisio_")
        zf.extractall(extract_dir)

    # If the ZIP contains a single top-level directory, descend into it so
    # the pipeline sees the repo root rather than a wrapper folder.
    entries = list(Path(extract_dir).iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return str(entries[0])

    return extract_dir


# ---------------------------------------------------------------------------
# Pipeline execution helper
# ---------------------------------------------------------------------------

def _execute_pipeline(target: str) -> tuple[bool, str]:
    """Run pipeline.run_pipeline(target) and capture all output.

    Returns:
        (success, log_text)
    """
    import pipeline  # lazy import; src/ is on sys.path

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            pipeline.run_pipeline(target.strip())
        return True, buf.getvalue()
    except SystemExit:
        return False, buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        buf.write(f"\nUnexpected error: {exc}")
        return False, buf.getvalue()


# ---------------------------------------------------------------------------
# HTML / CSS helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_COLOUR: dict[str, str] = {
    "high":   "#22c55e",
    "medium": "#eab308",
    "low":    "#52525b",
}


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _confidence_badge(level: str) -> str:
    c = _CONFIDENCE_COLOUR.get(level, "#52525b")
    return (
        f'<span style="background:{c}18;color:{c};border:1px solid {c}44;'
        f'border-radius:3px;padding:1px 7px;font-size:11px;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.07em;">{level}</span>'
    )


def _keyword_pill(word: str) -> str:
    return (
        f'<span style="background:rgba(255,255,255,0.04);color:#71717a;'
        f'border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:20px;padding:1px 9px;font-size:12px;margin:2px 3px;'
        f"display:inline-block;font-family:'JetBrains Mono',monospace;\">"
        f"{word}</span>"
    )


def _section(title: str) -> None:
    st.markdown(f"### {title}")


def _empty_state(msg: str) -> None:
    st.markdown(
        f'<p style="color:#8b949e;font-size:13px;padding:0.5rem 0;">{msg}</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab content renderers
# (function names kept stable; only user-facing labels changed)
# ---------------------------------------------------------------------------

def render_mapper(data: dict | None) -> None:
    """Tab 1 — Project Architecture."""
    if data is None:
        st.warning("No architecture data available. Run the pipeline first.")
        return

    col_ts, col_root = st.columns([3, 4])
    with col_ts:
        st.caption(f"Analysed  {data.get('generated_at', '—')}")
    with col_root:
        st.caption(f"Repo root  `{data.get('repo_root', '.')}`")

    st.divider()

    # ── Setup ────────────────────────────────────────────────────────────────
    _section("Environment Setup")
    setup = data.get("setup", {})
    lang  = setup.get("language") or "unknown"
    deps  = setup.get("dependencies_file")
    steps = setup.get("run_steps") or []

    if lang == "unknown" and not deps and not steps:
        _empty_state("Language and setup steps were not auto-detected for this repository.")
    else:
        lines = [f"**Language:** {lang}"]
        if deps:
            lines.append(f"**Dependencies:** `{deps}`")
        if steps:
            lines.append("**Run steps:**")
            lines.extend(f"  - `{s}`" for s in steps)
        st.markdown("\n\n".join(lines))

    st.divider()

    # ── Entry Points ─────────────────────────────────────────────────────────
    _section("Key Entry Points")
    entry_points: list[dict] = data.get("entry_points", [])

    if not entry_points:
        _empty_state("No entry points detected.")
    else:
        cols_per_row = 3
        for i in range(0, len(entry_points), cols_per_row):
            row  = entry_points[i : i + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, ep in zip(cols, row):
                with col:
                    st.metric(
                        label=ep.get("file", "—"),
                        value=ep.get("import_count", 0),
                        help=f"Import count — {ep.get('file')}",
                    )
                    st.markdown(
                        _confidence_badge(ep.get("confidence", "low")),
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── File breakdown ────────────────────────────────────────────────────────
    _section("File Breakdown")
    classification: dict = data.get("classification", {})
    if classification:
        rows = [
            {"File": f, "Category": v.get("category", "—"), "Reason": v.get("reason", "—")}
            for f, v in classification.items()
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        _empty_state("No classification data available.")


def render_mentor(data: dict | None) -> None:
    """Tab 2 — AI Codebase Insights."""
    if data is None:
        st.warning("No AI insights available. Run the pipeline first.")
        return

    meta = data.get("session_metadata", {})
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Questions answered", meta.get("total_questions", "—"))
    with m2:
        st.metric("Generated", meta.get("saved_at", "—"))

    st.divider()

    entries: list[dict] = data.get("entries", [])
    if not entries:
        _empty_state("No insights found.")
        return

    _section("Codebase Q&A")

    for idx, entry in enumerate(entries, start=1):
        question   = entry.get("question", "")
        answer     = entry.get("answer", "")
        confidence = entry.get("developer_confidence", "—")
        timestamp  = entry.get("timestamp", "—")

        st.markdown(
            f'<div style="margin-bottom:1.75rem;background:rgba(255,255,255,0.02);'
            f'border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:1.1rem 1.3rem;">'

            f'<p style="color:#3f3f46;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.09em;margin-bottom:0.55rem;font-weight:600;">'
            f'Insight {idx}</p>'

            f'<div style="display:flex;gap:0.75rem;margin-bottom:0.65rem;">'
            f'<span style="color:#e4e4e7;font-weight:700;font-size:13px;'
            f'min-width:1.4rem;padding-top:1px;flex-shrink:0;">Q</span>'
            f'<p style="color:#d4d4d8;font-size:13px;margin:0;line-height:1.65;'
            f'font-weight:500;">{question}</p>'
            f'</div>'

            f'<div style="display:flex;gap:0.75rem;border-left:2px solid rgba(255,255,255,0.07);'
            f'margin-left:0.6rem;padding-left:0.85rem;">'
            f'<span style="color:#71717a;font-weight:700;font-size:13px;'
            f'min-width:1.4rem;padding-top:1px;flex-shrink:0;">A</span>'
            f'<div style="color:#71717a;font-size:13px;line-height:1.75;'
            f'white-space:pre-wrap;overflow-wrap:break-word;flex:1;">{answer}</div>'
            f'</div>'

            f'<p style="color:#27272a;font-size:11px;margin-top:0.6rem;'
            f'padding-left:2.5rem;">confidence: {confidence}&nbsp;&middot;&nbsp;{timestamp}</p>'

            f'</div>',
            unsafe_allow_html=True,
        )


def render_coach(data: dict | None) -> None:
    """Tab 3 — Onboarding Setup."""
    if data is None:
        st.warning("No onboarding tasks available. Run the pipeline first.")
        return

    # ── Recommended starting point ────────────────────────────────────────────
    rec        = data.get("recommended_task") or {}
    rec_file   = rec.get("file", "—")
    rec_reason = rec.get("reason", "")

    _section("Where to Start")
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);'
        f'backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);'
        f'border:1px solid rgba(255,255,255,0.07);'
        f'border-left:3px solid #7c3aed;border-radius:0 10px 10px 0;'
        f'padding:1rem 1.25rem;margin-bottom:1rem;">'
        f'<p style="color:#e4e4e7;font-weight:600;font-size:14px;'
        f"font-family:'JetBrains Mono',monospace;margin:0 0 0.4rem;\">"
        f'{rec_file}</p>'
        f'<p style="color:#71717a;font-size:13px;margin:0;line-height:1.65;">'
        f'{rec_reason}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Self-assessment ───────────────────────────────────────────────────────
    quiz: list[dict] = data.get("quiz", [])
    count = len(quiz)
    _section(f"Self-Assessment  —  {count} question{'s' if count != 1 else ''}")

    if not quiz:
        _empty_state("No assessment questions generated yet.")
    else:
        for i, item in enumerate(quiz, start=1):
            topic    = item.get("topic", f"Question {i}")
            question = item.get("question", "")
            keywords: list[str] = item.get("expected_keywords", [])
            verdict  = item.get("verdict", "")

            with st.expander(f"Q{i} — {topic}", expanded=True):
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03);'
                    f'border-left:2px solid rgba(124,58,237,0.4);'
                    f'padding:0.65rem 1rem;border-radius:0 8px 8px 0;'
                    f'color:#d4d4d8;font-size:13px;line-height:1.65;'
                    f'margin-bottom:1rem;">{question}</div>',
                    unsafe_allow_html=True,
                )

                st.text_area(
                    "Your answer",
                    value="",
                    placeholder="Write your explanation here…",
                    height=110,
                    label_visibility="visible",
                    key=f"quiz_ans_{i}",
                )

                if keywords:
                    pills = "".join(_keyword_pill(kw) for kw in keywords)
                    st.markdown(
                        f'<div style="margin-top:0.6rem;">'
                        f'<span style="color:#3f3f46;font-size:11px;'
                        f'text-transform:uppercase;letter-spacing:0.06em;'
                        f'margin-right:6px;">Expected topics</span>{pills}</div>',
                        unsafe_allow_html=True,
                    )

                if verdict:
                    st.caption(f"Status: {verdict}")

    st.divider()

    topic_scores: dict = data.get("topic_scores", {})
    if topic_scores:
        _section("Understanding Summary")
        rows = [{"Topic": t, "Score": s} for t, s in topic_scores.items()]
        st.dataframe(rows, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# View: landing
# ---------------------------------------------------------------------------

def _view_landing() -> None:
    # Inject the animated oscillating background directly onto the app container.
    # Because Streamlit's <style> tag is global we target stAppViewContainer here.
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #050505 0%, #1a0b2e 40%, #0d1b2a 70%, #050505 100%) !important;
            background-size: 200% 200% !important;
            animation: gradientMove 15s ease infinite !important;
        }
        /* Remove the ::before radial glow on landing so it doesn't fight the gradient */
        [data-testid="stAppViewContainer"]::before { display: none !important; }
        /* Landing page: remove default block-container padding so the flex stage fills the viewport */
        [data-testid="block-container"] {
            padding: 0 !important;
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # All hero content lives inside .landing-stage — a full-viewport flex column.
    # The CTA button is rendered by st.button so state transitions still work;
    # it is wrapped in .landing-cta-wrap for the enlarged + pulsing CSS override.
    st.markdown(
        '<div class="landing-stage">'
        '<span class="landing-eyebrow">Developer Onboarding Copilot</span>'
        '<h1 class="landing-title">Your Developer<br>Onboarding Copilot</h1>'
        '<p class="landing-subtitle">'
        'Turn any codebase into a clear, personalised developer onboarding experience. '
        'Understand the architecture, get AI-powered insights, and know exactly where to start.'
        '</p>'
        '<div class="landing-cta-wrap" id="landing-cta-anchor"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Render the native Streamlit button — CSS in .landing-cta-wrap targets it.
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        if st.button("Start Trial Now  →", type="primary", use_container_width=True):
            _goto("input")


# ---------------------------------------------------------------------------
# View: input
# ---------------------------------------------------------------------------

def _view_input() -> None:
    st.markdown(
        '<div class="fade-in-up">'
        '<h1 style="margin-bottom:0.15rem;">Analyse a Repository</h1>'
        '<p style="color:#52525b;font-size:13px;margin-bottom:2.5rem;">'
        'Connect a public GitHub URL or upload a local ZIP archive to get started.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_url, col_sep, col_zip = st.columns([10, 1, 10])

    # ── Option A: GitHub URL ──────────────────────────────────────────────────
    with col_url:
        st.markdown(
            '<div class="option-card">'
            '<p class="option-card-title">🔗 Connect Repository</p>'
            '<p class="option-card-sub">Paste a public GitHub or Git URL</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        url_input = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/owner/repo",
            label_visibility="collapsed",
            key="url_input",
        )
        url_ready = bool(url_input.strip())
        if st.button(
            "Analyse URL",
            type="primary",
            use_container_width=True,
            disabled=not url_ready,
            key="btn_url",
        ):
            st.session_state.target = url_input.strip()
            st.session_state.pipeline_error = ""
            _goto("analyzing")

    # ── Separator ─────────────────────────────────────────────────────────────
    with col_sep:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:100%;min-height:180px;">'
            '<span style="color:#3f3f46;font-size:12px;font-weight:500;">or</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Option B: ZIP upload ──────────────────────────────────────────────────
    with col_zip:
        st.markdown(
            '<div class="option-card">'
            '<p class="option-card-title">📁 Upload Local Repository</p>'
            '<p class="option-card-sub">Upload a ZIP archive of your project</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "ZIP archive",
            type=["zip"],
            label_visibility="collapsed",
            key="zip_upload",
        )
        if st.button(
            "Analyse ZIP",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
            key="btn_zip",
        ):
            try:
                extracted_path = _extract_zip(uploaded)
                st.session_state.target = extracted_path
                st.session_state.pipeline_error = ""
                _goto("analyzing")
            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP archive.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to extract ZIP: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back", key="btn_back_input"):
        _goto("landing")


# ---------------------------------------------------------------------------
# View: analyzing
# ---------------------------------------------------------------------------

def _view_analyzing() -> None:
    target = st.session_state.target

    st.markdown(
        '<div class="analyzing-wrap">'
        '<p class="analyzing-title">Analysing your repository…</p>'
        '<p class="analyzing-sub">This may take a minute while the AI agents map your codebase.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.status("Running Provisio pipeline…", expanded=True) as status:
        st.write(f"Target: `{target}`")
        st.write("Mapping architecture and entry points…")

        # Real pipeline execution — no artificial delays
        success, log = _execute_pipeline(target)

        if log.strip():
            st.code(log.strip(), language=None)

        if success:
            status.update(
                label="Analysis complete  ✓",
                state="complete",
                expanded=False,
            )
            st.session_state.pipeline_error = ""
            _goto("dashboard")
        else:
            status.update(
                label="Pipeline encountered an error.",
                state="error",
                expanded=True,
            )
            st.session_state.pipeline_error = log

    # Reached only on failure (success path calls st.rerun via _goto).
    if st.session_state.pipeline_error:
        st.error(
            "The pipeline exited with an error. "
            "Check the log above for details."
        )

    if st.button("← Try Again", key="btn_back_analyzing"):
        _goto("input")


# ---------------------------------------------------------------------------
# View: dashboard
# ---------------------------------------------------------------------------

def _view_dashboard() -> None:
    # ── Header bar with gradient title + subtle Start Over button ─────────────
    target_display = st.session_state.target or "—"
    st.markdown(
        f'<div class="dash-header">'
        f'<div>'
        f'<p class="dash-header-title">Your Developer Onboarding Guide is Ready ✦</p>'
        f'<p class="dash-header-sub">{target_display}</p>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Start Over button — sits inline below the header, right-aligned
    _, reset_col = st.columns([8, 2])
    with reset_col:
        if st.button("↺ Start Over", type="secondary", use_container_width=True):
            _reset()

    # Load (or reload) fresh data
    mapper_data = load_mapper()
    mentor_data = load_mentor()
    coach_data  = load_coach()

    tab1, tab2, tab3 = st.tabs([
        "📁 Project Architecture",
        "🤖 AI Codebase Insights",
        "⚙️ Onboarding Setup",
    ])

    with tab1:
        render_mapper(mapper_data)

    with tab2:
        render_mentor(mentor_data)

    with tab3:
        render_coach(coach_data)


# ---------------------------------------------------------------------------
# Router / main
# ---------------------------------------------------------------------------

def main() -> None:
    _inject_css()
    _init_state()

    step = st.session_state.step

    if step == "landing":
        _view_landing()
    elif step == "input":
        _view_input()
    elif step == "analyzing":
        _view_analyzing()
    elif step == "dashboard":
        _view_dashboard()
    else:
        # Fallback — unknown state; reset cleanly
        _reset()


main()
