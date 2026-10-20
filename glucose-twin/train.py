"""Train the spike model:  python train.py --data data  [--kind T2DM]"""
import argparse, json, os
import joblib, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from pipeline import build_dataset


def best_thr(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    return float(th[np.argmax((2 * pr * rc / (pr + rc + 1e-9))[:-1])])

def report(y, p, thr):
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "Precision": precision_score(y, p >= thr), "Recall": recall_score(y, p >= thr), "F1": f1_score(y, p >= thr)}

def split(df, seed=42):
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(df, groups=df.patient))
    tv, te = df.iloc[a], df.iloc[b]
    c, d = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(tv, groups=tv.patient))
    return tv.iloc[c], tv.iloc[d], te            # train / validation / test: different patients in each


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--kind", default="T2DM")
    a = ap.parse_args()
    full, feats = build_dataset(a.data, a.kind)
    df = full.dropna(subset=["spike"])
    tr, va, te = split(df)
    print(f"patients train/val/test: {tr.patient.nunique()}/{va.patient.nunique()}/{te.patient.nunique()}  spike rate: {df.spike.mean():.2%}")
    try:
        import lightgbm as lgb
        backend = "LightGBM"
        model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
                                   scale_pos_weight=(tr.spike == 0).sum() / max((tr.spike == 1).sum(), 1))
        model.fit(tr[feats], tr.spike, eval_set=[(va[feats], va.spike)], callbacks=[lgb.early_stopping(50, verbose=False)])
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        backend = "scikit-learn HistGradientBoosting (install lightgbm for the full model)"
        model = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, class_weight="balanced", random_state=42)
        model.fit(tr[feats], tr.spike)
    pv, pt = model.predict_proba(va[feats])[:, 1], model.predict_proba(te[feats])[:, 1]
    thr = best_thr(va.spike, pv)
    bthr = best_thr(va.spike, va.g_last)                       # baseline: "predict spike if glucose is already high"
    metrics = {"Baseline (current glucose only)": report(te.spike, te.g_last.values, bthr), "Twin model": report(te.spike, pt, thr)}
    if hasattr(model, "booster_"):
        imp = model.booster_.feature_importance("gain")
    else:
        from sklearn.inspection import permutation_importance
        s = te.sample(min(2000, len(te)), random_state=0)
        imp = permutation_importance(model, s[feats], s.spike, n_repeats=3, random_state=0).importances_mean
    imp = np.clip(imp, 0, None) / max(np.sum(np.clip(imp, 0, None)), 1e-9)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib"); joblib.dump(full, "models/featured.joblib")
    json.dump({"features": feats, "threshold": thr, "metrics": metrics, "backend": backend, "kind": a.kind,
               "importance": dict(sorted(zip(feats, map(float, imp)), key=lambda kv: -kv[1])[:15]),
               "test_patients": sorted(te.patient.unique().tolist()), "n_patients": int(df.patient.nunique()),
               "n_rows": int(len(df)), "spike_rate": float(df.spike.mean())}, open("models/meta.json", "w"), indent=2)
    for k, v in metrics.items():
        print(k, {m: round(x, 3) for m, x in v.items()})
