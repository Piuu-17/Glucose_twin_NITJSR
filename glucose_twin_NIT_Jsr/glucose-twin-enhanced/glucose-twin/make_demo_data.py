"""Creates SYNTHETIC data in the same layout as the Shanghai files, only to test the app. Not real patients."""
import os, numpy as np, pandas as pd
rng, out = np.random.default_rng(0), "data_demo"
os.makedirs(f"{out}/Shanghai_T2DM", exist_ok=True)
rows = []
for k in range(30):
    rid, a1c = f"{2000 + k}_0_20210701", float(np.clip(rng.normal(8, 1.5), 5.5, 13))
    rows.append({"Patient Number": rid, "Type of Diabetes": "T2DM", "Sex": rng.choice(["Male", "Female"]),
                 "Age (years)": int(rng.integers(40, 80)), "BMI (kg/m2)": round(rng.normal(25, 3.5), 1),
                 "Duration of Diabetes (years)": int(rng.integers(1, 25)), "HbA1c (mmol/mol)": round(a1c * 10.9 - 23.5, 1)})
    t = pd.date_range("2021-07-01", periods=96 * 5, freq="15min")
    g, cho, idx = 100 + (a1c - 5.5) * 14 + rng.normal(0, 6, len(t)), np.zeros(len(t)), np.arange(len(t))
    for d in range(5):
        for h in (7.5, 12.5, 18.5):
            i = int(d * 96 + h * 4); cho[i] = rng.integers(40, 90)
            x = np.clip((idx - i) / 4, 0, None) / 1.2
            g += np.where(idx >= i, cho[i] * (0.9 + (a1c - 5.5) * 0.12) * x * np.exp(1 - x), 0)
    g = pd.Series(g).rolling(3, min_periods=1).mean().round(1)
    pd.DataFrame({"Date": t, "CGM (mg / dl)": g, "CHO (g)": np.where(cho > 0, cho, np.nan),
                  "CSII - bolus insulin (Novolin R, IU)": np.where(cho > 0, cho / 10, np.nan)}
                 ).to_csv(f"{out}/Shanghai_T2DM/{rid}.csv", index=False)
pd.DataFrame(rows).to_csv(f"{out}/Shanghai_T2DM_Summary.csv", index=False)
print("Wrote", out)
