"""Loading, labelling and feature engineering for the Shanghai T2DM / T1DM CGM datasets."""
import glob, os, re
import numpy as np, pandas as pd

HORIZON, THRESH = 8, 180.0   # 8 readings x 15 min = 2 h ahead; spike = glucose > 180 mg/dL


def _read(p):
    return pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p)

def _find(cols, *keys):
    return next((c for c in cols if any(k in str(c).lower() for k in keys)), None)

def _clean(c):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(c)).strip("_")


def load_summary(data_dir, kind):
    pats = glob.glob(os.path.join(data_dir, f"*{kind}*Summary*")) or glob.glob(os.path.join(data_dir, "*Summary*"))
    pats = [p for p in pats if p.lower().endswith((".xlsx", ".xls", ".csv"))]
    if not pats:
        raise FileNotFoundError(f"No *Summary* file found in {data_dir}")
    df = _read(sorted(pats)[0])
    df = df.rename(columns={_find(df.columns, "patient") or df.columns[0]: "record_id"})
    df["record_id"] = df["record_id"].astype(str).str.strip()
    return df


def clean_summary(df):
    """Numeric columns are kept; small text columns become one-hot flags; the rest is dropped."""
    out = pd.DataFrame({"record_id": df["record_id"]})
    for c in df.columns:
        if c == "record_id":
            continue
        s, num = df[c], pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() >= 0.6:
            out["p_" + _clean(c)] = num
        elif 1 < s.nunique() <= 8:
            d = pd.get_dummies(s.fillna("NA").astype(str), prefix="p_" + _clean(c)).astype(int)
            d.columns = [_clean(x) for x in d.columns]
            out = pd.concat([out, d], axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def load_cgm(data_dir, kind):
    d = next((p for p in sorted(glob.glob(os.path.join(data_dir, "*")))
              if os.path.isdir(p) and kind.lower() in os.path.basename(p).lower()), None)
    if d is None:
        raise FileNotFoundError(f"No folder containing '{kind}' found in {data_dir}")
    frames = []
    for f in sorted(glob.glob(os.path.join(d, "*"))):
        if not f.lower().endswith((".csv", ".xlsx", ".xls")) or os.path.basename(f).startswith("~$"):
            continue
        df = _read(f)
        tc, gc = _find(df.columns, "date", "time"), _find(df.columns, "cgm")
        if tc is None or gc is None:
            print(f"Skipping {os.path.basename(f)}: no date or CGM column"); continue
        mc = [c for c in df.columns if "dietary" in str(c).lower() or "饮食" in str(c)]   # meal notes (text) mark meal times
        ic = [c for c in df.columns if "insulin" in str(c).lower() and "non-insulin" not in str(c).lower() and "basal" not in str(c).lower()]
        x = pd.DataFrame({
            "glucose": pd.to_numeric(df[gc], errors="coerce"),
            "meal": df[mc].notna().any(axis=1).astype(float) if mc else 0.0,
            "insulin": df[ic].apply(pd.to_numeric, errors="coerce").sum(axis=1) if ic else 0.0,
        }).set_index(pd.to_datetime(df[tc], errors="coerce"))
        x = x[x.index.notna()]
        x.index.name = "timestamp"
        x = x.resample("15min").agg({"glucose": "mean", "meal": "sum", "insulin": "sum"})
        x["glucose"] = x["glucose"].interpolate(limit=2, limit_area="inside")   # fill gaps of <= 30 min only
        x["record_id"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(x.reset_index())
    if not frames:
        raise RuntimeError("No usable CGM files found.")
    cgm = pd.concat(frames, ignore_index=True)
    if cgm["glucose"].median() < 35:          # values look like mmol/L
        cgm["glucose"] *= 18.0
    cgm = cgm[(cgm["glucose"].isna()) | cgm["glucose"].between(40, 500)]
    ok = cgm["glucose"].notna()
    cgm["seg"] = (ok != ok.groupby(cgm["record_id"]).shift()).groupby(cgm["record_id"]).cumsum()  # split at gaps
    return cgm[ok].reset_index(drop=True)


def _features(g):
    g, s = g.copy(), g["glucose"]
    g["g_last"] = s
    for k in (1, 2, 4):
        g[f"diff_{k * 15}m"] = s - s.shift(k)
    for w in (4, 8, 16):                       # 1 h, 2 h, 4 h of history
        r = s.rolling(w)
        g[f"mean_{w}"], g[f"std_{w}"], g[f"max_{w}"], g[f"min_{w}"] = r.mean(), r.std(), r.max(), r.min()
    g["meals_2h"] = (g["meal"] > 0).astype(float).rolling(8, min_periods=1).sum()
    last = g["timestamp"].where(g["meal"] > 0).ffill()
    g["since_meal_h"] = ((g["timestamp"] - last).dt.total_seconds() / 3600).fillna(6).clip(upper=6)   # hours since last meal, capped at 6
    g["insulin_2h"] = g["insulin"].rolling(8, min_periods=1).sum()
    hr = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60
    g["hour_sin"], g["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    fut = s[::-1].rolling(HORIZON).max()[::-1].shift(-1)       # max of the NEXT 8 readings (never the current one)
    g["future_max"] = fut
    g["spike"] = np.where(fut.isna(), np.nan, (fut > THRESH).astype(float))
    return g


def build_dataset(data_dir, kind="T2DM"):
    sm = clean_summary(load_summary(data_dir, kind))
    cgm = load_cgm(data_dir, kind)
    ids, first = set(sm.record_id), {i.split("_")[0]: i for i in sm.record_id}
    cgm["sid"] = cgm.record_id.map(lambda r: r if r in ids else first.get(r.split("_")[0]))
    print(f"{cgm.record_id.nunique()} recordings, {cgm.sid.notna().groupby(cgm.record_id).first().sum()} matched to the summary file")
    df = pd.concat([_features(g) for _, g in cgm.groupby(["record_id", "seg"], sort=False)], ignore_index=True)
    df = df.dropna(subset=["std_16"]).merge(sm.rename(columns={"record_id": "sid"}), on="sid", how="left")
    df["patient"] = df.record_id.str.split("_").str[0]
    dyn = [c for c in df.columns if c.split("_")[0] in ("g", "diff", "mean", "std", "max", "min", "meals", "since", "insulin", "hour")
           and c not in ("future_max",)]
    static = [c for c in df.columns if c.startswith("p_") and df[c].notna().any()]
    return df, dyn + static
