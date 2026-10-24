"""Glucose Twin: run with  streamlit run app.py  (after python train.py --data data)"""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st

M = Path(__file__).parent / "models"

# ── palette: purple / white / blue ────────────────────────────────────────────────────────────────
P900, P700, P600, P500, P300, P100 = "#2E1065", "#6D28D9", "#7C3AED", "#8B5CF6", "#C4B5FD", "#EDE9FE"
B700, B600, B500, B400, B100 = "#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#DBEAFE"
INK, MUTED, GRID, ACCENT = "#1E1B4B", "#6B6A8F", "#EEE9FF", "#C026D3"
UP, DOWN = P600, B500                       # drivers that raise / lower risk
FONT = "Plus Jakarta Sans, Inter, sans-serif"

NICE = {"g_last": "Glucose now", "diff_15m": "Change, 15 min", "diff_30m": "Change, 30 min", "diff_60m": "Change, 1 h",
        "meals_2h": "Meals, last 2 h", "since_meal_h": "Hours since meal", "insulin_2h": "Insulin, last 2 h", "hour_sin": "Time of day", "hour_cos": "Time of day"}
def nice(f):
    if f in NICE: return NICE[f]
    for p, n in (("mean_", "Average"), ("std_", "Variability"), ("max_", "Peak"), ("min_", "Lowest")):
        if f.startswith(p): return f"{n}, last {int(f[len(p):]) * 15 // 60} h"
    return f.replace("p_", "", 1).replace("_", " ")

def html(s):
    """Markdown treats indented lines as code, so flatten the markup to one line."""
    st.markdown("".join(l.strip() for l in s.splitlines()), unsafe_allow_html=True)

def show(fig, **kw):
    """Plotly chart that works on old and new Streamlit versions."""
    cfg = {"displayModeBar": False}
    try: st.plotly_chart(fig, width="stretch", theme=None, config=cfg, **kw)
    except TypeError: st.plotly_chart(fig, use_container_width=True, theme=None, config=cfg, **kw)

def card(name):
    """Bordered container; the key gives it a CSS class (.st-key-card_*) on recent Streamlit versions."""
    try: return st.container(border=True, key=f"card_{name}")
    except TypeError: return st.container(border=True)

def blend(n, a=P600, b=B500):
    """n colours fading from a to b (hex)."""
    A, B = [np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for c in (a, b)]
    return ["#%02x%02x%02x" % tuple((A + (B - A) * t).astype(int)) for t in np.linspace(0, 1, max(n, 1))]

def base_layout(fig, h, **kw):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK, size=12),
                      hoverlabel=dict(bgcolor="#fff", bordercolor=P300, font=dict(family=FONT, color=INK)), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=P100, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=P100, tickfont=dict(color=MUTED))
    return fig

st.set_page_config(page_title="Glucose Twin", page_icon="🩸", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {--p900:#2E1065; --p700:#6D28D9; --p600:#7C3AED; --p300:#C4B5FD; --p100:#EDE9FE; --b600:#2563EB; --b500:#3B82F6; --b100:#DBEAFE;
       --ink:#1E1B4B; --muted:#6B6A8F; --line:#E4DEFF; --shadow:0 1px 2px rgba(76,29,149,.06), 0 8px 24px rgba(76,29,149,.07);}

/* ── canvas ── */
.stApp {font-family:'Plus Jakarta Sans', sans-serif;
        background: radial-gradient(1100px 520px at 88% -8%, rgba(96,165,250,.20), transparent 60%),
                    radial-gradient(900px 480px at -5% 0%, rgba(167,139,250,.22), transparent 55%), #F7F5FF;}
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button, .stApp input, .stApp td, .stApp th {font-family:'Plus Jakarta Sans', sans-serif;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px;}
header[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer {visibility: hidden;}
::selection {background: var(--p300); color: var(--p900);}
h2, h3 {color: var(--p900); font-weight: 800; letter-spacing: -.015em;}

/* ── hero ── */
.hero {position:relative; overflow:hidden; border-radius:22px; padding:1.7rem 2rem 1.5rem; color:#fff; margin-bottom:1.3rem;
       background: linear-gradient(118deg, #4C1D95 0%, #6D28D9 42%, #2563EB 100%); box-shadow: 0 18px 40px rgba(76,29,149,.28);}
.hero::before {content:""; position:absolute; width:340px; height:340px; right:-60px; top:-150px; border-radius:50%;
               background: radial-gradient(circle, rgba(255,255,255,.28), transparent 68%);}
.hero::after {content:""; position:absolute; width:260px; height:260px; left:34%; bottom:-170px; border-radius:50%;
              background: radial-gradient(circle, rgba(147,197,253,.38), transparent 70%);}
.hero > * {position:relative; z-index:1;}
.hero-row {display:flex; align-items:center; gap:1rem;}
.hero h1 {color:#fff; margin:0; padding:0; font-size:2.15rem; font-weight:800; letter-spacing:-.025em; line-height:1.1;}
.hero p.sub {margin:.25rem 0 0; color:rgba(255,255,255,.82); font-size:.98rem;}
.hero-pills {margin-top:1.05rem; display:flex; flex-wrap:wrap; gap:.5rem;}
.hero-pills span {background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); backdrop-filter: blur(6px);
                  border-radius:999px; padding:.22rem .8rem; font-size:.78rem; font-weight:600; letter-spacing:.01em;}

/* ── sidebar ── */
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #FFFFFF 0%, #F5F2FF 100%); border-right: 1px solid var(--line);}
.brand {display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.1rem;}
.brand b {display:block; font-size:1.08rem; color:var(--p900); font-weight:800; letter-spacing:-.01em; line-height:1.1;}
.brand span {font-size:.74rem; color:var(--muted); font-weight:500;}
.side-label {font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--p600); margin:1rem 0 .3rem;}
.side-note {font-size:.74rem; color:var(--muted); line-height:1.45; margin-top:1.4rem; padding:.7rem .85rem; background:#fff; border:1px dashed var(--p300); border-radius:12px;}
[data-testid="stExpander"] {background:#fff; border:1px solid var(--line) !important; border-radius:14px; box-shadow: var(--shadow);}
[data-testid="stExpander"] summary {font-weight:600; color:var(--p900);}

/* ── tabs: pill switcher ── */
.stTabs [data-baseweb="tab-list"] {gap:4px; background:#fff; border:1px solid var(--line); border-radius:999px; padding:5px; width:fit-content; box-shadow: var(--shadow);}
.stTabs [data-baseweb="tab"] {height:40px; border-radius:999px; padding:0 20px; background:transparent;}
.stTabs [data-baseweb="tab"] p {font-weight:700; color:var(--muted); font-size:.92rem;}
.stTabs [aria-selected="true"] {background: linear-gradient(120deg, #7C3AED, #2563EB);}
.stTabs [aria-selected="true"] p {color:#fff;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
.stTabs [data-baseweb="tab-panel"] {padding-top:1.1rem;}

/* ── cards (bordered containers) ── */
[class*="st-key-card_"], div[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border:1px solid var(--line) !important; border-radius:20px !important;
        box-shadow: var(--shadow); animation: rise .55s cubic-bezier(.2,.7,.2,1) both;}
[class*="st-key-card_"] [class*="st-key-card_"] {box-shadow:none;}
.card-title {font-weight:800; font-size:1.02rem; color:var(--p900); letter-spacing:-.01em; margin:0;}
.card-sub {font-size:.8rem; color:var(--muted); margin:.1rem 0 .4rem;}

/* ── section headings, chips, KPI tiles ── */
.sec {display:flex; align-items:center; gap:.7rem; margin:1.1rem 0 .7rem;}
.sec h4 {margin:0; padding:0; font-size:.78rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; color:var(--p700);}
.sec::after {content:""; flex:1; height:1px; background: linear-gradient(90deg, var(--p300), transparent);}
.pill {font-size:.68rem; font-weight:800; letter-spacing:.08em; padding:.15rem .6rem; border-radius:999px; color:#fff; background:linear-gradient(120deg,#7C3AED,#2563EB);}
.chip {display:inline-flex; align-items:center; gap:.4rem; background:#fff; border:1px solid var(--p300); color:var(--p900); border-radius:999px;
       padding:.2rem .85rem; margin:0 .45rem .55rem 0; font-size:.82rem; font-weight:600; box-shadow:0 1px 2px rgba(76,29,149,.06);}
.chip i {font-style:normal; color:var(--muted); font-weight:500;}
.kpis {display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:1rem; margin-bottom:1.2rem;}
.kpi {position:relative; overflow:hidden; background:#fff; border:1px solid var(--line); border-radius:18px; padding:1rem 1.15rem 1rem;
      box-shadow:var(--shadow); transition: transform .2s ease, box-shadow .2s ease; animation: rise .55s cubic-bezier(.2,.7,.2,1) both;}
.kpi:hover {transform: translateY(-3px); box-shadow:0 2px 4px rgba(76,29,149,.08), 0 16px 32px rgba(76,29,149,.14);}
.kpi::before {content:""; position:absolute; inset:0 0 auto 0; height:4px; background:linear-gradient(90deg, #7C3AED, #3B82F6);}
.kpi:nth-child(2) {animation-delay:.06s;} .kpi:nth-child(3) {animation-delay:.12s;} .kpi:nth-child(4) {animation-delay:.18s;}
.kpi-top {display:flex; align-items:center; justify-content:space-between;}
.kpi-label {font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--muted);}
.kpi-ico {width:30px; height:30px; display:grid; place-items:center; border-radius:10px; font-size:.95rem; background:linear-gradient(135deg, #EDE9FE, #DBEAFE);}
.kpi-value {font-size:2.05rem; font-weight:800; color:var(--p900); letter-spacing:-.03em; line-height:1.15; margin-top:.35rem;}
.kpi-value small {font-size:.78rem; font-weight:600; color:var(--muted); margin-left:.35rem; letter-spacing:0;}
.kpi-sub {font-size:.78rem; font-weight:600; margin-top:.15rem; color:var(--b600);}
.kpi-sub.up {color:var(--p700);}

/* ── verdict banner ── */
.verdict {border-radius:16px; padding:1rem 1.25rem; color:#fff; margin:.2rem 0 .3rem; position:relative; overflow:hidden;}
.verdict::after {content:""; position:absolute; width:180px; height:180px; right:-50px; top:-80px; border-radius:50%; background:radial-gradient(circle, rgba(255,255,255,.3), transparent 70%);}
.verdict.hi {background:linear-gradient(120deg, #5B21B6, #9333EA 60%, #C026D3); animation: glow 2.4s ease-in-out infinite;}
.verdict.lo {background:linear-gradient(120deg, #1D4ED8, #3B82F6 60%, #60A5FA);}
.verdict small {display:block; font-size:.68rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; opacity:.85;}
.verdict b {display:block; font-size:1.4rem; font-weight:800; letter-spacing:-.015em; margin-top:.15rem;}
.verdict span {font-size:.82rem; opacity:.9;}

.note {font-size:.82rem; color:var(--muted); background:#fff; border:1px solid var(--line); border-left:4px solid var(--p600); border-radius:12px; padding:.65rem .95rem; margin-top:.5rem;}
.disclaimer {margin-top:1rem; font-size:.82rem; color:#3730A3; background:linear-gradient(120deg, #EDE9FE, #DBEAFE); border-radius:14px; padding:.8rem 1.1rem;}

/* ── widgets ── */
[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:14px; overflow:hidden;}
div[data-baseweb="select"] > div {border-radius:12px; background:#fff;}
[data-testid="stSlider"] [role="slider"] {box-shadow:0 0 0 4px rgba(124,58,237,.18);}
[data-testid="stAlert"] {border-radius:14px;}

@keyframes rise {from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:none;}}
@keyframes glow {0%,100% {box-shadow:0 8px 22px rgba(124,58,237,.30);} 50% {box-shadow:0 8px 34px rgba(192,38,211,.50);}}
@media (max-width: 640px) {.hero {padding:1.2rem 1.1rem;} .hero h1 {font-size:1.6rem;} .kpi-value {font-size:1.7rem;}}
</style>""", unsafe_allow_html=True)

LOGO = ('<svg width="{s}" height="{s}" viewBox="0 0 44 44"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7C3AED"/>'
        '<stop offset="1" stop-color="#2563EB"/></linearGradient></defs><rect width="44" height="44" rx="13" fill="{bg}"/>'
        '<path d="M22 9c-4.5 6-9 10.2-9 15.2a9 9 0 0018 0C31 19.2 26.5 15 22 9z" fill="{drop}"/>'
        '<path d="M15 25h4l2-4 3 7 2-3h3" stroke="{line}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')

if not (M / "model.joblib").exists():
    st.error("No trained model found. Run `python train.py --data data` first, then reload this page."); st.stop()

@st.cache_resource
def load():
    return joblib.load(M / "model.joblib"), json.load(open(M / "meta.json")), joblib.load(M / "featured.joblib")
model, meta, df = load()
feats, thr = meta["features"], meta["threshold"]

html(f"""<div class="hero"><div class="hero-row">{LOGO.format(s=52, bg="rgba(255,255,255,.18)", drop="#fff", line="#6D28D9")}
<div><h1>Glucose Twin</h1><p class="sub">A virtual patient that estimates whether glucose will go above 180 mg/dL in the next 2 hours.</p></div></div>
<div class="hero-pills"><span>⏱ 2-hour horizon</span><span>🎯 Spike &gt; 180 mg/dL</span><span>⚙ {meta["backend"].split(" (")[0]}</span>
<span>👥 {meta["n_patients"]} patients</span></div></div>""")

twin, rep = st.tabs(["🧬  Patient twin", "📊  Model report"])

with st.sidebar:
    html(f"""<div class="brand">{LOGO.format(s=38, bg="url(#lg)", drop="#fff", line="#6D28D9")}<div><b>Glucose Twin</b><span>Virtual patient explorer</span></div></div>
    <div class="side-label">Patient &amp; moment</div>""")
    held = set(meta["test_patients"])
    rid = st.selectbox("Recording", sorted(df.record_id.unique()),
                       format_func=lambda r: r + ("  (unseen in training)" if r.split("_")[0] in held else ""))
    p = df[df.record_id == rid].sort_values("timestamp").reset_index(drop=True)
    i = st.select_slider("Moment in recording", options=list(range(len(p))), value=len(p) // 2,
                         format_func=lambda k: p.timestamp[k].strftime("%d %b, %H:%M"))
    row = p.iloc[i]
    X = row[feats].astype(float).to_frame().T
    html('<div class="side-label">Scenario</div>')
    with st.expander("What if…", expanded=False):
        g = st.slider("Glucose now (mg/dL)", 40, 400, int(row.g_last), key=f"g{rid}{i}")
        d = st.slider("Change over last hour", -100, 100, int(np.nan_to_num(row.diff_60m)), key=f"d{rid}{i}")
        c = st.slider("Hours since last meal", 0.0, 6.0, float(row.since_meal_h), 0.25, key=f"c{rid}{i}")
        st.caption("Other inputs stay as recorded.")
    if g != int(row.g_last): X["g_last"] = g
    if d != int(np.nan_to_num(row.diff_60m)): X["diff_60m"] = d
    if c != float(row.since_meal_h): X["since_meal_h"] = c
    html('<div class="side-note">Research prototype. Not a medical device and not for treatment decisions.</div>')

whatif = bool(set(X.columns[(X.iloc[0] != row[feats].astype(float)) & ~(X.iloc[0].isna() & row[feats].astype(float).isna())]))
prob = float(model.predict_proba(X)[0, 1])
high = prob >= thr

with twin:
    # patient profile chips
    profile = [(k, row[k]) for k in row.index if k.startswith("p_") and pd.notna(row[k]) and any(s in k.lower() for s in ("age", "bmi", "duration", "hba1c"))]
    if profile:
        html("".join(f'<span class="chip"><i>{nice(k)}</i> {v:g}</span>' for k, v in profile[:5]))

    # snapshot KPIs (reflect the what-if values when used)
    gnow, dh = float(X.g_last.iloc[0]), float(np.nan_to_num(X.diff_60m.iloc[0]))
    since, ins = float(X.since_meal_h.iloc[0]), float(np.nan_to_num(X.insulin_2h.iloc[0])) if "insulin_2h" in X else 0.0
    status = "Below range" if gnow < 70 else ("In target range" if gnow <= 180 else "Above target")
    trend = ("▲ Rising" if dh > 5 else "▼ Falling" if dh < -5 else "▬ Steady")
    html(f'<div class="sec"><h4>Snapshot · {row.timestamp.strftime("%d %b, %H:%M")}</h4>{"<span class=pill>WHAT-IF ACTIVE</span>" if whatif else ""}</div>')
    html(f"""<div class="kpis">
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">Glucose now</div><div class="kpi-ico">🩸</div></div>
      <div class="kpi-value">{gnow:.0f}<small>mg/dL</small></div><div class="kpi-sub {'up' if gnow > 180 else ''}">{status}</div></div>
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">1-hour change</div><div class="kpi-ico">📈</div></div>
      <div class="kpi-value">{dh:+.0f}<small>mg/dL</small></div><div class="kpi-sub {'up' if dh > 5 else ''}">{trend}</div></div>
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">Since last meal</div><div class="kpi-ico">🍽️</div></div>
      <div class="kpi-value">{since:.1f}<small>hours</small></div><div class="kpi-sub">{'Recent meal' if since < 2 else 'Fasting window' if since >= 4 else 'Post-meal tail'}</div></div>
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">Insulin, last 2 h</div><div class="kpi-ico">💉</div></div>
      <div class="kpi-value">{ins:.1f}<small>units</small></div><div class="kpi-sub">{'No insulin logged' if ins == 0 else 'Dose logged'}</div></div></div>""")

    left, right = st.columns([3, 2], gap="large")

    with left, card("chart"):
        html('<p class="card-title">Glucose trace</p><p class="card-sub">Recorded history up to the selected moment, and what happened next.</p>')
        lo, hi = max(0, i - 24), min(len(p), i + 9)
        seg = p.glucose[lo:hi]
        t_now, t_end = p.timestamp[i], p.timestamp[min(i + 8, len(p) - 1)]
        fig = go.Figure()
        fig.add_hrect(y0=70, y1=180, fillcolor=B100, opacity=.38, line_width=0, layer="below")
        fig.add_vrect(x0=t_now, x1=t_end, fillcolor=P600, opacity=.07, line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=p.timestamp[lo:i + 1], y=p.glucose[lo:i + 1], name="Recorded so far", mode="lines", line=dict(color=P600, width=3.5, shape="spline", smoothing=.5),
                                 fill="tozeroy", fillcolor="rgba(124,58,237,.10)", hovertemplate="%{y:.0f} mg/dL"))
        fig.add_trace(go.Scatter(x=p.timestamp[i:hi], y=p.glucose[i:hi], name="What happened next", mode="lines", line=dict(color=B500, width=3, dash="dot", shape="spline", smoothing=.5),
                                 hovertemplate="%{y:.0f} mg/dL"))
        fig.add_trace(go.Scatter(x=[t_now], y=[row.glucose], name="Selected moment", mode="markers", marker=dict(size=13, color="#fff", line=dict(color=P700, width=3.5)),
                                 hovertemplate="Now: %{y:.0f} mg/dL"))
        fig.add_hline(y=180, line_color=ACCENT, line_dash="dash", line_width=1.6, annotation_text="180 mg/dL", annotation_font=dict(color=ACCENT, size=11), annotation_position="top left")
        fig.add_annotation(x=t_now, y=1, yref="paper", text="2 h forecast window →", showarrow=False, xanchor="left", yanchor="top", xshift=6,
                           font=dict(size=11, color=P700), bgcolor="rgba(255,255,255,.7)")
        base_layout(fig, 400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Glucose (mg/dL)", hovermode="x unified", legend=dict(orientation="h", y=1.13, x=0, font=dict(size=12)))
        fig.update_yaxes(range=[max(0, min(seg.min(), 70) - 15), max(seg.max(), 190) + 25])
        show(fig, key="trace")
        if pd.notna(row.spike):
            did = row.spike == 1
            html(f'<div class="note">In the recording, glucose <b>{"did" if did else "did not"}</b> go above 180 mg/dL in the following 2 hours.</div>')
        else:
            html('<div class="note">The recording ends within 2 hours of this moment, so the outcome is unknown.</div>')

    with right, card("risk"):
        html('<p class="card-title">Spike risk</p><p class="card-sub">Model estimate for the next 2 hours.</p>')
        html(f"""<div class="verdict {'hi' if high else 'lo'}"><small>2-hour outlook</small>
        <b>{'▲ Spike likely' if high else '✓ Spike unlikely'}</b>
        <span>Estimated probability {prob:.0%} · decision threshold {thr:.0%}</span></div>""")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number=dict(suffix="%", valueformat=".0f", font=dict(size=44, color=P900, family=FONT)),
                                       gauge=dict(axis=dict(range=[0, 100], tickwidth=0, tickcolor=P100, tickfont=dict(size=10, color=MUTED)),
                                                  bar=dict(color=P600 if high else B500, thickness=.34), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                                  steps=[dict(range=[0, thr * 100], color=B100), dict(range=[thr * 100, 100], color=P100)],
                                                  threshold=dict(line=dict(color=ACCENT, width=4), thickness=.9, value=thr * 100))))
        base_layout(gauge, 210, margin=dict(l=20, r=20, t=20, b=0))
        show(gauge, key="gauge")
        if hasattr(model, "booster_"):
            s = pd.Series(model.booster_.predict(X, pred_contrib=True)[0][:-1], index=feats)
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:7][::-1]
            html('<p class="card-title" style="margin-top:.2rem">Why this estimate</p><p class="card-sub">Purple pushes risk up, blue pushes it down.</p>')
            f2 = go.Figure(go.Bar(x=s.values, y=[nice(k) for k in s.index], orientation="h", marker=dict(color=[UP if v > 0 else DOWN for v in s.values], cornerradius=6),
                                  hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
            base_layout(f2, 300, margin=dict(l=0, r=0, t=4, b=0), bargap=.35)
            f2.update_xaxes(zeroline=True, zerolinecolor=P300, zerolinewidth=1.5)
            f2.update_yaxes(gridcolor="rgba(0,0,0,0)")
            show(f2, key="why")

with rep:
    met = meta["metrics"]
    m = pd.DataFrame(met).round(3)
    tw, bl = met.get("Twin model", {}), met.get("Baseline (current glucose only)", {})
    gain = tw.get("PR-AUC", 0) - bl.get("PR-AUC", 0)
    html(f"""<div class="kpis">
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">Patients</div><div class="kpi-ico">👥</div></div><div class="kpi-value">{meta['n_patients']}</div><div class="kpi-sub">Split by patient, no overlap</div></div>
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">CGM readings</div><div class="kpi-ico">📟</div></div><div class="kpi-value">{meta['n_rows']:,}</div><div class="kpi-sub">15-minute resolution</div></div>
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">Spike rate</div><div class="kpi-ico">⚡</div></div><div class="kpi-value">{meta['spike_rate']:.0%}</div><div class="kpi-sub">Of moments followed by &gt;180</div></div>
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">Twin PR-AUC</div><div class="kpi-ico">🎯</div></div><div class="kpi-value">{tw.get('PR-AUC', float('nan')):.2f}</div><div class="kpi-sub {'up' if gain > 0 else ''}">{gain:+.2f} vs baseline</div></div></div>""")

    c1, c2 = st.columns(2, gap="large")
    with c1, card("metrics"):
        html(f'<p class="card-title">Results on patients the model never saw</p><p class="card-sub">Model: {meta["backend"]}</p>')
        fm = go.Figure()
        for (name, col), colr in zip(m.items(), (P300 if len(m.columns) > 1 else P600, P600)):
            fm.add_trace(go.Bar(x=m.index, y=col.values, name=name, marker=dict(color=colr if name == "Twin model" else B400, cornerradius=6), hovertemplate="%{x}: %{y:.3f}"))
        base_layout(fm, 300, barmode="group", bargap=.3, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15, x=0))
        fm.update_yaxes(range=[0, 1.05]); fm.update_xaxes(gridcolor="rgba(0,0,0,0)")
        show(fm, key="metrics")
        try: st.dataframe(m.style.format("{:.3f}").highlight_max(axis=1, color="#DDD6FE"), width="stretch")
        except TypeError: st.dataframe(m.style.format("{:.3f}").highlight_max(axis=1, color="#DDD6FE"), use_container_width=True)
    with c2, card("importance"):
        html('<p class="card-title">What the model relies on</p><p class="card-sub">Share of total gain across the top features.</p>')
        imp = pd.Series(meta["importance"]).iloc[:12][::-1]
        fi = go.Figure(go.Bar(x=imp.values, y=[nice(k) for k in imp.index], orientation="h", marker=dict(color=blend(len(imp), B400, P600), cornerradius=6),
                              hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        base_layout(fi, 470, margin=dict(l=0, r=0, t=6, b=0), bargap=.3)
        fi.update_xaxes(tickformat=".0%"); fi.update_yaxes(gridcolor="rgba(0,0,0,0)")
        show(fi, key="importance")
    html("""<div class="disclaimer"><b>How to read this.</b> Spike = glucose above 180 mg/dL at any point in the next 2 hours. Train, validation and test sets contain different patients.
    The dataset is small and from one region, so treat these numbers as a research prototype, not a medical device.</div>""")
