"""Glucose Twin: run with  streamlit run app.py  (after python train.py --data data)"""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st

M = Path(__file__).parent / "models"
LOW, HIGH, INK, BLUE = "#2F7D6B", "#B8322A", "#12262D", "#1F5FBF"
NICE = {"g_last": "Glucose now", "diff_15m": "Change, 15 min", "diff_30m": "Change, 30 min", "diff_60m": "Change, 1 h",
        "meals_2h": "Meals, last 2 h", "since_meal_h": "Hours since meal", "insulin_2h": "Insulin, last 2 h", "hour_sin": "Time of day", "hour_cos": "Time of day"}
def nice(f):
    if f in NICE: return NICE[f]
    for p, n in (("mean_", "Average"), ("std_", "Variability"), ("max_", "Peak"), ("min_", "Lowest")):
        if f.startswith(p): return f"{n}, last {int(f[len(p):]) * 15 // 60} h"
    return f.replace("p_", "", 1).replace("_", " ")

st.set_page_config(page_title="Glucose Twin", page_icon="🩸", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;800&display=swap');
html, body, [class*="css"] {font-family: 'Public Sans', sans-serif;}
.block-container {padding-top: 2rem; max-width: 1200px;}
h1 {font-weight: 800; letter-spacing: -.02em;}
.risk {border-radius: 14px; padding: 1.2rem 1.5rem; color: #fff; margin-bottom: 1rem;}
.risk b {font-size: 3.2rem; display: block; line-height: 1.05;}
.chip {display: inline-block; border: 1px solid #b9cbd1; border-radius: 999px; padding: .15rem .8rem; margin: 0 .4rem .5rem 0; font-size: .85rem;}
</style>""", unsafe_allow_html=True)

if not (M / "model.joblib").exists():
    st.error("No trained model found. Run `python train.py --data data` first, then reload this page."); st.stop()

@st.cache_resource
def load():
    return joblib.load(M / "model.joblib"), json.load(open(M / "meta.json")), joblib.load(M / "featured.joblib")
model, meta, df = load()
feats, thr = meta["features"], meta["threshold"]

st.title("Glucose Twin")
st.caption("A virtual patient that estimates whether glucose will go above 180 mg/dL in the next 2 hours.")
twin, rep = st.tabs(["Patient twin", "Model report"])

with st.sidebar:
    held = set(meta["test_patients"])
    rid = st.selectbox("Recording", sorted(df.record_id.unique()),
                       format_func=lambda r: r + ("  (unseen in training)" if r.split("_")[0] in held else ""))
    p = df[df.record_id == rid].sort_values("timestamp").reset_index(drop=True)
    i = st.select_slider("Moment in recording", options=list(range(len(p))), value=len(p) // 2,
                         format_func=lambda k: p.timestamp[k].strftime("%d %b, %H:%M"))
    row = p.iloc[i]
    X = row[feats].astype(float).to_frame().T
    with st.expander("What if…"):
        g = st.slider("Glucose now (mg/dL)", 40, 400, int(row.g_last), key=f"g{rid}{i}")
        d = st.slider("Change over last hour", -100, 100, int(np.nan_to_num(row.diff_60m)), key=f"d{rid}{i}")
        c = st.slider("Hours since last meal", 0.0, 6.0, float(row.since_meal_h), 0.25, key=f"c{rid}{i}")
        st.caption("Other inputs stay as recorded.")
    if g != int(row.g_last): X["g_last"] = g
    if d != int(np.nan_to_num(row.diff_60m)): X["diff_60m"] = d
    if c != float(row.since_meal_h): X["since_meal_h"] = c

prob = float(model.predict_proba(X)[0, 1])
high = prob >= thr

with twin:
    profile = [(k, row[k]) for k in row.index if k.startswith("p_") and pd.notna(row[k]) and any(s in k.lower() for s in ("age", "bmi", "duration", "hba1c"))]
    st.markdown("".join(f'<span class="chip">{nice(k)}: {v:g}</span>' for k, v in profile[:5]), unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        lo, hi = max(0, i - 24), min(len(p), i + 9)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.timestamp[lo:i + 1], y=p.glucose[lo:i + 1], name="Recorded so far", line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(x=p.timestamp[i:hi], y=p.glucose[i:hi], name="What happened next", line=dict(color="#7C8C92", width=2, dash="dot")))
        fig.add_hline(y=180, line_color=HIGH, line_dash="dash", annotation_text="180 mg/dL")
        fig.add_vrect(x0=p.timestamp[i], x1=p.timestamp[min(i + 8, len(p) - 1)], fillcolor=BLUE, opacity=.07, line_width=0)
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Glucose (mg/dL)", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12), font=dict(color=INK))
        st.plotly_chart(fig, use_container_width=True)
        if pd.notna(row.spike):
            st.caption("In the recording, glucose " + ("did" if row.spike == 1 else "did not") + " go above 180 mg/dL in the following 2 hours.")
        else:
            st.caption("The recording ends within 2 hours of this moment, so the outcome is unknown.")

    with right:
        st.markdown(f'<div class="risk" style="background:{HIGH if high else LOW}"><b>{prob:.0%}</b>'
                    f'{"Spike likely in the next 2 hours" if high else "Spike unlikely in the next 2 hours"}</div>', unsafe_allow_html=True)
        if hasattr(model, "booster_"):
            s = pd.Series(model.booster_.predict(X, pred_contrib=True)[0][:-1], index=feats)
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:7][::-1]
            f2 = go.Figure(go.Bar(x=s.values, y=[nice(k) for k in s.index], orientation="h", marker_color=[HIGH if v > 0 else LOW for v in s.values]))
            f2.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title="Why: red pushes risk up, green pushes it down",
                             title_font_size=14, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK))
            st.plotly_chart(f2, use_container_width=True)

with rep:
    m = pd.DataFrame(meta["metrics"]).round(3)
    c1, c2 = st.columns(2)
    c1.subheader("Results on patients the model never saw"); c1.dataframe(m, use_container_width=True)
    c1.caption(f"{meta['n_patients']} patients, {meta['n_rows']:,} readings, spike rate {meta['spike_rate']:.0%}. Model: {meta['backend']}.")
    c2.subheader("What the model relies on")
    imp = pd.Series(meta["importance"]).iloc[:12][::-1]
    c2.plotly_chart(go.Figure(go.Bar(x=imp.values, y=[nice(k) for k in imp.index], orientation="h", marker_color=BLUE)).update_layout(
        height=360, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), use_container_width=True)
    st.info("Spike = glucose above 180 mg/dL at any point in the next 2 hours. Train, validation and test sets contain different patients. "
            "The dataset is small and from one region, so treat these numbers as a research prototype, not a medical device.")
