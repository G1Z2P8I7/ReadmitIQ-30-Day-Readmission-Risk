"""ReadmitIQ Dashboard Theme & Styling System.

Provides CSS injection, styled HTML metric cards, and helper functions
for the dark-mode glassmorphism clinical dashboard design.
"""

# ---------------------------------------------------------------------------
# Color palette constants
# ---------------------------------------------------------------------------
PRIMARY_DARK = "#0B1B3F"
CARD_BG = "rgba(255, 255, 255, 0.05)"
CARD_BORDER = "rgba(255, 255, 255, 0.08)"
ACCENT_TEAL = "#00D4AA"
ACCENT_CORAL = "#FF6B6B"
ACCENT_GOLD = "#FFB800"
ACCENT_BLUE = "#4DA8FF"
TEXT_WHITE = "#F0F2F6"
TEXT_MUTED = "#8892B0"
SURFACE_LIGHT = "#112240"


def inject_css() -> str:
    """Return the full CSS stylesheet for the dark glassmorphism theme."""
    return f"""
<style>
/* ── Global Overrides ─────────────────────────────────── */
.stApp {{
    background: linear-gradient(135deg, {PRIMARY_DARK} 0%, #0a192f 50%, #0d2137 100%);
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%) !important;
    border-right: 1px solid {CARD_BORDER};
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {{
    color: {TEXT_MUTED} !important;
    font-size: 0.88rem;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {TEXT_WHITE} !important;
}}

/* Radio / navigation pills */
div[data-testid="stRadio"] label {{
    color: {TEXT_MUTED} !important;
    transition: color 0.2s;
}}
div[data-testid="stRadio"] label:hover {{
    color: {ACCENT_TEAL} !important;
}}

/* ── Glass Card ───────────────────────────────────────── */
.glass-card {{
    background: {CARD_BG};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.glass-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 212, 170, 0.10);
}}

/* ── Metric Card ──────────────────────────────────────── */
.metric-card {{
    background: {CARD_BG};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0, 212, 170, 0.12);
}}
.metric-icon {{
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 0.75rem;
}}
.metric-value {{
    font-size: 2rem;
    font-weight: 800;
    color: {TEXT_WHITE};
    line-height: 1.2;
    margin-bottom: 0.25rem;
}}
.metric-label {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.metric-delta {{
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.35rem;
}}
.delta-positive {{ color: {ACCENT_TEAL}; }}
.delta-negative {{ color: {ACCENT_CORAL}; }}
.delta-neutral {{ color: {ACCENT_GOLD}; }}

/* ── Hero Section ─────────────────────────────────────── */
.hero-title {{
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, {TEXT_WHITE} 0%, {ACCENT_TEAL} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.15;
}}
.hero-subtitle {{
    font-size: 1.1rem;
    color: {TEXT_MUTED};
    font-weight: 400;
    max-width: 720px;
    line-height: 1.6;
}}

/* ── Section Headers ──────────────────────────────────── */
.section-header {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {TEXT_WHITE};
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.section-sub {{
    font-size: 0.9rem;
    color: {TEXT_MUTED};
    margin-bottom: 1.25rem;
}}

/* ── Gradient Divider ─────────────────────────────────── */
.gradient-divider {{
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%,
        {ACCENT_TEAL}40 20%,
        {ACCENT_TEAL}80 50%,
        {ACCENT_TEAL}40 80%,
        transparent 100%
    );
    border: none;
    margin: 1.5rem 0;
}}

/* ── Safeguard / Feature Cards ────────────────────────── */
.safeguard-card {{
    background: {CARD_BG};
    backdrop-filter: blur(12px);
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
    padding: 1.25rem;
    height: 100%;
}}
.safeguard-card h4 {{
    color: {ACCENT_TEAL};
    font-size: 1rem;
    margin-bottom: 0.75rem;
}}
.safeguard-card p, .safeguard-card li {{
    color: {TEXT_MUTED};
    font-size: 0.88rem;
    line-height: 1.6;
}}

/* ── Risk Band Badges ─────────────────────────────────── */
.risk-badge {{
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
}}
.risk-high {{
    background: rgba(255, 107, 107, 0.15);
    color: {ACCENT_CORAL};
    border: 1px solid rgba(255, 107, 107, 0.3);
}}
.risk-moderate {{
    background: rgba(255, 184, 0, 0.15);
    color: {ACCENT_GOLD};
    border: 1px solid rgba(255, 184, 0, 0.3);
}}
.risk-low {{
    background: rgba(0, 212, 170, 0.15);
    color: {ACCENT_TEAL};
    border: 1px solid rgba(0, 212, 170, 0.3);
}}

/* ── Alert / Info Banners ─────────────────────────────── */
.info-banner {{
    background: rgba(77, 168, 255, 0.08);
    border: 1px solid rgba(77, 168, 255, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: {TEXT_MUTED};
    font-size: 0.9rem;
    line-height: 1.6;
}}
.info-banner strong {{
    color: {ACCENT_BLUE};
}}
.warning-banner {{
    background: rgba(255, 184, 0, 0.08);
    border: 1px solid rgba(255, 184, 0, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: {TEXT_MUTED};
    font-size: 0.9rem;
    line-height: 1.6;
}}
.warning-banner strong {{
    color: {ACCENT_GOLD};
}}
.danger-banner {{
    background: rgba(255, 107, 107, 0.08);
    border: 1px solid rgba(255, 107, 107, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: {TEXT_MUTED};
    font-size: 0.9rem;
    line-height: 1.6;
}}
.danger-banner strong {{
    color: {ACCENT_CORAL};
}}
.success-banner {{
    background: rgba(0, 212, 170, 0.08);
    border: 1px solid rgba(0, 212, 170, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: {TEXT_MUTED};
    font-size: 0.9rem;
    line-height: 1.6;
}}
.success-banner strong {{
    color: {ACCENT_TEAL};
}}

/* ── Streamlit dataframe override ─────────────────────── */
.stDataFrame {{
    border-radius: 12px;
    overflow: hidden;
}}

/* ── Tabs styling ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    background: {CARD_BG};
    border-radius: 10px;
    border: 1px solid {CARD_BORDER};
    color: {TEXT_MUTED};
    padding: 0.5rem 1rem;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(0, 212, 170, 0.12) !important;
    border-color: {ACCENT_TEAL} !important;
    color: {ACCENT_TEAL} !important;
}}

/* ── Form styling ─────────────────────────────────────── */
.stForm {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 1.5rem;
}}

/* ── Image containers ─────────────────────────────────── */
.stImage {{
    border-radius: 12px;
    overflow: hidden;
}}

/* ── Primary model highlight row ──────────────────────── */
.primary-model-tag {{
    background: rgba(0, 212, 170, 0.15);
    color: {ACCENT_TEAL};
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}}
</style>
"""


# ---------------------------------------------------------------------------
# HTML component helpers
# ---------------------------------------------------------------------------


def metric_card(
    icon: str,
    value: str,
    label: str,
    color: str = ACCENT_TEAL,
    delta: str = "",
    delta_type: str = "positive",
) -> str:
    """Return HTML for a styled glassmorphism metric card."""
    delta_html = ""
    if delta:
        css_class = {
            "positive": "delta-positive",
            "negative": "delta-negative",
            "neutral": "delta-neutral",
        }.get(delta_type, "delta-neutral")
        delta_html = f'<div class="metric-delta {css_class}">{delta}</div>'

    return f"""
    <div class="metric-card">
        <div class="metric-icon" style="background: {color}20; color: {color};">
            {icon}
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def hero_section(title: str, subtitle: str) -> str:
    """Return HTML for the hero section with gradient title."""
    return f"""
    <div style="margin-bottom: 2rem;">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """


def section_header(icon: str, title: str, subtitle: str = "") -> str:
    """Return HTML for a styled section header."""
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="section-header">{icon} {title}</div>
    {sub_html}
    """


def gradient_divider() -> str:
    """Return HTML for a gradient divider line."""
    return '<div class="gradient-divider"></div>'


def glass_card(content: str) -> str:
    """Wrap content in a glass-card container."""
    return f'<div class="glass-card">{content}</div>'


def safeguard_card(icon: str, title: str, bullets: list[str]) -> str:
    """Return HTML for a safeguard/feature card with icon and bullet points."""
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f"""
    <div class="safeguard-card">
        <h4>{icon} {title}</h4>
        <ul style="padding-left: 1.2rem; margin: 0;">{items}</ul>
    </div>
    """


def risk_badge(band: str) -> str:
    """Return HTML for a colored risk band badge."""
    band_lower = band.lower()
    if "high" in band_lower:
        css = "risk-high"
    elif "moderate" in band_lower or "medium" in band_lower:
        css = "risk-moderate"
    else:
        css = "risk-low"
    return f'<span class="risk-badge {css}">{band}</span>'


def info_banner(content: str) -> str:
    """Return HTML for a blue info banner."""
    return f'<div class="info-banner">{content}</div>'


def warning_banner(content: str) -> str:
    """Return HTML for a gold warning banner."""
    return f'<div class="warning-banner">{content}</div>'


def danger_banner(content: str) -> str:
    """Return HTML for a coral danger banner."""
    return f'<div class="danger-banner">{content}</div>'


def success_banner(content: str) -> str:
    """Return HTML for a teal success banner."""
    return f'<div class="success-banner">{content}</div>'
