"""ReadmitIQ Theme & Styling System — Function Health Aesthetic.

Provides CSS injection and HTML component generators inspired by Function Health's
editorial luxury healthcare design: warm alabaster canvas, terracotta accents,
editorial serif headings with italicized emphasis, and warm sand cards.
"""

# ---------------------------------------------------------------------------
# Color palette constants
# ---------------------------------------------------------------------------
CANVAS_BG = "#FAF8F5"  # Warm Alabaster / Cream
CARD_BG = "#F3EFE6"  # Soft Sand / Oat
CARD_BORDER = "#E5DFD3"  # Warm Stone Border
CARD_HOVER_BORDER = "#D6CDBC"

TEXT_PRIMARY = "#1A1715"  # Deep Espresso Charcoal
TEXT_SECONDARY = "#5C564F"  # Muted Taupe Charcoal
TEXT_MUTED = "#8A8276"  # Soft Warm Taupe

ACCENT_TERRACOTTA = "#A84B29"  # Function Brand Burnt Sienna
ACCENT_TERRACOTTA_LIGHT = "#F2E8E3"
ACCENT_TERRACOTTA_HOVER = "#8E3D20"

ACCENT_SAGE = "#3B6E53"  # In-Range / Favorable Green
ACCENT_SAGE_LIGHT = "#E9F1EC"

ACCENT_AMBER = "#C07D2B"  # Moderate / Advisory Gold
ACCENT_AMBER_LIGHT = "#FBF4E8"

ACCENT_ROSE = "#B84033"  # High Risk Alert Red
ACCENT_ROSE_LIGHT = "#FCEEEB"


def inject_css() -> str:
    """Return the full CSS stylesheet for the Function Health aesthetic."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Global Canvas ────────────────────────────────────────── */
html, body, [class*="st-emotion"], .stApp {{
    background-color: {CANVAS_BG} !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* ── Typography ───────────────────────────────────────────── */
h1, h2, h3, h4, .fh-editorial {{
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 500 !important;
    color: {TEXT_PRIMARY} !important;
    letter-spacing: -0.02em !important;
}}

.fh-italic {{
    font-family: 'Newsreader', Georgia, serif !important;
    font-style: italic !important;
    font-weight: 400 !important;
    color: {ACCENT_TERRACOTTA} !important;
}}

/* ── Streamlit Top Header & Toolbar ──────────────────────── */
header[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: #F5F1E8 !important;
    border-right: 1px solid {CARD_BORDER} !important;
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.88rem !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {TEXT_PRIMARY} !important;
}}

/* Radio navigation */
div[data-testid="stRadio"] {{
    background: transparent;
}}
div[data-testid="stRadio"] > label {{
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    color: {TEXT_PRIMARY} !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label {{
    background: #FAF8F5 !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 9999px !important;
    padding: 0.5rem 1rem !important;
    margin-bottom: 0.4rem !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
    border-color: {ACCENT_TERRACOTTA} !important;
    background: {ACCENT_TERRACOTTA_LIGHT} !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {{
    background: {ACCENT_TERRACOTTA} !important;
    border-color: {ACCENT_TERRACOTTA} !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p {{
    color: #FFFFFF !important;
    font-weight: 600 !important;
}}

/* ── Hero Section ─────────────────────────────────────────── */
.fh-hero {{
    text-align: center;
    padding: 2.5rem 1rem 2rem 1rem;
    max-width: 960px;
    margin: 0 auto 1.5rem auto;
}}
.fh-hero-title {{
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 3.2rem !important;
    font-weight: 500 !important;
    color: {TEXT_PRIMARY} !important;
    line-height: 1.15 !important;
    margin-bottom: 0.75rem !important;
    letter-spacing: -0.025em !important;
}}
.fh-hero-sub {{
    font-size: 1.12rem !important;
    color: {TEXT_SECONDARY} !important;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.65 !important;
    font-weight: 400 !important;
}}

/* ── Section Titles ───────────────────────────────────────── */
.fh-section-title {{
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 2.1rem !important;
    font-weight: 500 !important;
    color: {TEXT_PRIMARY} !important;
    line-height: 1.25 !important;
    margin-bottom: 0.35rem !important;
    letter-spacing: -0.02em !important;
}}
.fh-section-sub {{
    font-size: 0.95rem !important;
    color: {TEXT_MUTED} !important;
    margin-bottom: 1.25rem !important;
    line-height: 1.5 !important;
}}

/* ── Metric Cards ─────────────────────────────────────────── */
.fh-metric-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 20px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    height: 100%;
}}
.fh-metric-card:hover {{
    transform: translateY(-2px);
    border-color: {CARD_HOVER_BORDER};
    box-shadow: 0 8px 24px rgba(26, 23, 21, 0.04);
}}
.fh-metric-icon {{
    width: 44px;
    height: 44px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 0.75rem;
    background: #FAF8F5;
    border: 1px solid {CARD_BORDER};
}}
.fh-metric-val {{
    font-size: 2.25rem;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    line-height: 1.15;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}}
.fh-metric-lbl {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.fh-metric-badge {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.5rem;
}}
.fh-badge-pos {{
    background: {ACCENT_SAGE_LIGHT};
    color: {ACCENT_SAGE};
}}
.fh-badge-terra {{
    background: {ACCENT_TERRACOTTA_LIGHT};
    color: {ACCENT_TERRACOTTA};
}}
.fh-badge-amber {{
    background: {ACCENT_AMBER_LIGHT};
    color: {ACCENT_AMBER};
}}

/* ── Step Cards (01, 02, 03 from Function Health) ────────── */
.fh-step-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 22px;
    padding: 1.75rem 1.5rem;
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.fh-step-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(26, 23, 21, 0.04);
}}
.fh-step-num {{
    font-size: 0.9rem;
    font-weight: 700;
    color: {ACCENT_TERRACOTTA};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}}
.fh-step-title {{
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 1.5rem !important;
    font-weight: 500 !important;
    color: {TEXT_PRIMARY} !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.25 !important;
}}
.fh-step-sub {{
    font-size: 0.88rem;
    color: {TEXT_MUTED};
    margin-bottom: 1rem;
    line-height: 1.5;
}}
.fh-step-item {{
    background: #FAF8F5;
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 0.65rem 0.9rem;
    font-size: 0.85rem;
    color: {TEXT_PRIMARY};
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* ── Warm Banners ─────────────────────────────────────────── */
.fh-banner {{
    border-radius: 16px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 1.25rem;
    font-size: 0.92rem;
    line-height: 1.6;
}}
.fh-banner-terra {{
    background: {ACCENT_TERRACOTTA_LIGHT};
    border: 1px solid rgba(168, 75, 41, 0.2);
    color: #5A2310;
}}
.fh-banner-terra strong {{
    color: {ACCENT_TERRACOTTA};
}}
.fh-banner-amber {{
    background: {ACCENT_AMBER_LIGHT};
    border: 1px solid rgba(192, 125, 43, 0.2);
    color: #613B0E;
}}
.fh-banner-amber strong {{
    color: {ACCENT_AMBER};
}}
.fh-banner-sage {{
    background: {ACCENT_SAGE_LIGHT};
    border: 1px solid rgba(59, 110, 83, 0.2);
    color: #1A3E2C;
}}
.fh-banner-sage strong {{
    color: {ACCENT_SAGE};
}}
.fh-banner-rose {{
    background: {ACCENT_ROSE_LIGHT};
    border: 1px solid rgba(184, 64, 51, 0.2);
    color: #611812;
}}
.fh-banner-rose strong {{
    color: {ACCENT_ROSE};
}}

/* ── Result Gauge Card (Function Test Style) ──────────────── */
.fh-result-container {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 24px;
    padding: 2rem;
    margin: 1.5rem 0;
}}
.fh-gauge-track {{
    background: #E5DFD3;
    height: 10px;
    border-radius: 9999px;
    position: relative;
    margin: 1.5rem 0 1rem 0;
}}
.fh-gauge-zone-normal {{
    position: absolute;
    left: 0;
    width: 20%;
    height: 100%;
    background: {ACCENT_SAGE};
    border-radius: 9999px 0 0 9999px;
    opacity: 0.7;
}}
.fh-gauge-zone-elevated {{
    position: absolute;
    left: 20%;
    right: 0;
    height: 100%;
    background: {ACCENT_TERRACOTTA};
    border-radius: 0 9999px 9999px 0;
    opacity: 0.7;
}}

/* ── Form Inputs Override ─────────────────────────────────── */
.stForm {{
    background: {CARD_BG} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 24px !important;
    padding: 2rem !important;
}}
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput div[data-baseweb="input"] > div,
.stTextInput div[data-baseweb="input"] > div {{
    background-color: #FAF8F5 !important;
    border-color: {CARD_BORDER} !important;
    border-radius: 12px !important;
    color: {TEXT_PRIMARY} !important;
}}
.stSlider {{
    padding: 0.5rem 0 !important;
}}

/* Form submit button pill */
button[kind="primaryFormSubmit"],
.stButton > button {{
    background-color: {ACCENT_TERRACOTTA} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(168, 75, 41, 0.25) !important;
}}
button[kind="primaryFormSubmit"]:hover,
.stButton > button:hover {{
    background-color: {ACCENT_TERRACOTTA_HOVER} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(168, 75, 41, 0.35) !important;
}}

/* ── Tabs Override ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    background: transparent !important;
    border-bottom: 1px solid {CARD_BORDER} !important;
    padding-bottom: 0.5rem !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 9999px !important;
    border: 1px solid transparent !important;
    color: {TEXT_MUTED} !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT_PRIMARY} !important;
    background: {CARD_BG} !important;
}}
.stTabs [aria-selected="true"] {{
    background: {ACCENT_TERRACOTTA_LIGHT} !important;
    border-color: rgba(168, 75, 41, 0.3) !important;
    color: {ACCENT_TERRACOTTA} !important;
    font-weight: 600 !important;
}}

/* ── Dataframes ───────────────────────────────────────────── */
.stDataFrame {{
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid {CARD_BORDER} !important;
}}

/* ── Warm Divider ─────────────────────────────────────────── */
.fh-divider {{
    height: 1px;
    background: {CARD_BORDER};
    border: none;
    margin: 2rem 0;
}}

/* ── Image Frames ─────────────────────────────────────────── */
.stImage {{
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid {CARD_BORDER} !important;
    background: #FFFFFF !important;
    padding: 0.5rem !important;
}}
</style>
"""


# ---------------------------------------------------------------------------
# HTML Component Generators
# ---------------------------------------------------------------------------


def hero_section(main_title: str, italic_part: str, subtitle: str) -> str:
    """Return Function Health style editorial hero header."""
    return (
        f'<div class="fh-hero">'
        f'<h1 class="fh-hero-title">{main_title} <span class="fh-italic">{italic_part}</span></h1>'
        f'<p class="fh-hero-sub">{subtitle}</p>'
        f"</div>"
    )


def editorial_header(main_title: str, italic_part: str, subtitle: str = "") -> str:
    """Return an editorial section header."""
    sub_html = f'<div class="fh-section-sub">{subtitle}</div>' if subtitle else ""
    return (
        f'<div style="margin-bottom: 1.25rem;">'
        f'<h2 class="fh-section-title">{main_title} <span class="fh-italic">{italic_part}</span></h2>'
        f"{sub_html}"
        f"</div>"
    )


def metric_card(
    icon: str,
    value: str,
    label: str,
    delta: str = "",
    delta_type: str = "pos",
    subtext: str = "",
) -> str:
    """Return a warm sand Function Health metric card."""
    badge_html = ""
    if delta:
        badge_class = {
            "pos": "fh-badge-pos",
            "terra": "fh-badge-terra",
            "amber": "fh-badge-amber",
        }.get(delta_type, "fh-badge-pos")
        badge_html = f'<div class="fh-metric-badge {badge_class}">{delta}</div>'

    sub_html = (
        f'<div style="font-size:0.75rem; color:{TEXT_MUTED}; margin-top:0.3rem;">{subtext}</div>'
        if subtext
        else ""
    )

    return (
        f'<div class="fh-metric-card">'
        f'<div class="fh-metric-icon">{icon}</div>'
        f'<div class="fh-metric-val">{value}</div>'
        f'<div class="fh-metric-lbl">{label}</div>'
        f"{badge_html}"
        f"{sub_html}"
        f"</div>"
    )


def step_card(
    number_str: str, title: str, italic_word: str, subtitle: str, bullets: list[str]
) -> str:
    """Return a Function Health 01/02/03 step card."""
    items = "".join(
        f'<div class="fh-step-item"><span>✓</span> <span>{b}</span></div>' for b in bullets
    )
    return (
        f'<div class="fh-step-card">'
        f'<div class="fh-step-num">{number_str}</div>'
        f'<div class="fh-step-title">{title} <span class="fh-italic">{italic_word}</span></div>'
        f'<div class="fh-step-sub">{subtitle}</div>'
        f"<div>{items}</div>"
        f"</div>"
    )


def risk_gauge_card(
    prob_pct: float,
    mult: float,
    band: str,
    cutoff_pct: float = 10.77,
) -> str:
    """Return Function Health test result format with visual in-range / elevated gauge."""
    is_elevated = prob_pct >= cutoff_pct
    status_label = "Elevated Risk (Top 20% Capacity)" if is_elevated else "Standard Discharge Risk"
    status_class = "fh-badge-terra" if is_elevated else "fh-badge-pos"
    marker_pos = min(max(prob_pct / 30.0 * 100.0, 5.0), 95.0)
    marker_color = ACCENT_TERRACOTTA if is_elevated else ACCENT_SAGE

    return (
        f'<div class="fh-result-container">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">'
        f"<div>"
        f'<div style="font-size:0.8rem; font-weight:700; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">'
        f"Readmission Risk Assessment"
        f"</div>"
        f"<div style=\"font-family:'Newsreader', Georgia, serif; font-size:2.8rem; font-weight:500; color:{TEXT_PRIMARY}; line-height:1.1; margin:0.3rem 0;\">"
        f"{prob_pct:.1f}% <span style=\"font-family:'Plus Jakarta Sans', sans-serif; font-size:1.1rem; font-weight:600; color:{TEXT_SECONDARY};\">({mult:.2f}× hospital baseline)</span>"
        f"</div>"
        f"</div>"
        f"<div>"
        f'<span class="fh-metric-badge {status_class}" style="font-size:0.88rem; padding:0.4rem 1rem;">'
        f"{band} · {status_label}"
        f"</span>"
        f"</div>"
        f"</div>"
        f'<div class="fh-gauge-track">'
        f'<div class="fh-gauge-zone-normal"></div>'
        f'<div class="fh-gauge-zone-elevated"></div>'
        f'<div style="position:absolute; left:calc({marker_pos}% - 7px); top:-4px; width:18px; height:18px; border-radius:50%; background:{marker_color}; border:3px solid #FFFFFF; box-shadow:0 2px 6px rgba(0,0,0,0.25);"></div>'
        f"</div>"
        f'<div style="display:flex; justify-content:space-between; font-size:0.75rem; color:{TEXT_MUTED}; font-weight:600; text-transform:uppercase;">'
        f"<span>Standard Care (&lt; 10.8%)</span>"
        f'<span style="color:{ACCENT_TERRACOTTA};">Top 20% Capacity Tier (≥ 10.8%)</span>'
        f"</div>"
        f"</div>"
    )


def info_banner(text: str) -> str:
    return f'<div class="fh-banner fh-banner-terra">{text}</div>'


def warning_banner(text: str) -> str:
    return f'<div class="fh-banner fh-banner-amber">{text}</div>'


def success_banner(text: str) -> str:
    return f'<div class="fh-banner fh-banner-sage">{text}</div>'


def danger_banner(text: str) -> str:
    return f'<div class="fh-banner fh-banner-rose">{text}</div>'


def divider() -> str:
    return '<hr class="fh-divider"/>'


def sidebar_brand() -> str:
    return (
        f'<div style="padding: 1.25rem 0.5rem 1rem 0.5rem;">'
        f"<div style=\"font-family:'Newsreader', Georgia, serif; font-size:1.85rem; font-weight:500; color:{TEXT_PRIMARY}; letter-spacing:-0.02em;\">"
        f'Readmit<span style="font-style:italic; color:{ACCENT_TERRACOTTA};">IQ</span>'
        f"</div>"
        f'<div style="font-size:0.8rem; color:{TEXT_SECONDARY}; margin-top:0.2rem;">'
        f"Clinical Decision Support &amp; Capacity Prioritization"
        f"</div>"
        f"</div>"
    )


def sidebar_section_label(text: str) -> str:
    return (
        f'<div style="font-size:0.75rem; font-weight:700; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">'
        f"{text}"
        f"</div>"
    )


def form_section_header(title: str) -> str:
    return (
        f"<div style=\"font-family:'Newsreader', Georgia, serif; font-size:1.3rem; font-weight:500; color:{TEXT_PRIMARY}; margin-bottom:0.75rem;\">"
        f"{title}"
        f"</div>"
    )


def case_card_header(title: str, subtitle: str) -> str:
    return (
        f'<div style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:18px; padding:1.2rem; margin-bottom:1rem;">'
        f"<div style=\"font-family:'Newsreader', Georgia, serif; font-size:1.3rem; font-weight:500; color:{TEXT_PRIMARY};\">"
        f"{title}"
        f"</div>"
        f'<div style="font-size:0.8rem; color:{TEXT_SECONDARY}; margin-top:0.15rem;">'
        f"{subtitle}"
        f"</div>"
        f"</div>"
    )


def governance_card() -> str:
    return (
        f'<div style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:20px; padding:1.75rem;">'
        f"<div style=\"font-family:'Newsreader', Georgia, serif; font-size:1.5rem; font-weight:500; color:{TEXT_PRIMARY}; margin-bottom:0.75rem;\">"
        f"Clinical AI Governance &amp; Fairness Invariants"
        f"</div>"
        f'<ul style="color:{TEXT_SECONDARY}; line-height:1.8; font-size:0.92rem; padding-left:1.25rem;">'
        f"<li><strong>Protected Attributes Policy:</strong> Race and gender are strictly excluded from predictive model features and evaluated solely for disparity audits.</li>"
        f'<li><strong>Post-Processing Mitigation:</strong> Fairlearn <code>ThresholdOptimizer(constraints="equalized_odds", objective="balanced_accuracy_score", prefit=True)</code> adjusts decision boundaries per sensitive group.</li>'
        f"<li><strong>Small Subgroup Alert:</strong> Subgroups with N &lt; 500 (Asian, Other, Hispanic) carry wider bootstrap uncertainty intervals and are explicitly flagged.</li>"
        f"<li><strong>Non-Causal Usage:</strong> Model predictions reflect statistical correlations at discharge to support outreach capacity, never to decide treatment.</li>"
        f"</ul>"
        f"</div>"
    )


def rationale_card() -> str:
    return (
        f'<div style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:20px; padding:1.75rem;">'
        f"<div style=\"font-family:'Newsreader', Georgia, serif; font-size:1.4rem; font-weight:500; color:{TEXT_PRIMARY}; margin-bottom:0.5rem;\">"
        f"Primary Model Selection Rationale"
        f"</div>"
        f'<ul style="color:{TEXT_SECONDARY}; line-height:1.75; font-size:0.92rem; padding-left:1.25rem; margin-bottom:0;">'
        f"<li><strong>Superior Clinical Discrimination:</strong> Calibrated XGBoost delivers <strong>0.1385 PR-AUC</strong> (+54.1% over prevalence floor) and <strong>0.6344 ROC-AUC</strong>, exceeding simple prior-inpatient heuristics by 2.4× in clinical lift.</li>"
        f"<li><strong>Empirical Risk Calibration:</strong> Post-hoc isotonic calibration reduces Expected Calibration Error to <strong>0.0062</strong> with a Brier score of <strong>0.0806</strong>, ensuring predicted probabilities directly mirror actual readmission rates.</li>"
        f"<li><strong>Capacity-Constrained Efficiency:</strong> Flagging the top 20% of discharged patients captures <strong>34.53% of all 30-day readmissions</strong> with a 1.73× lift.</li>"
        f"</ul>"
        f"</div>"
    )
