import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(r"D:\PS84")
OUT = ROOT / "assets" / "training_curves"
OUT.mkdir(parents=True, exist_ok=True)
# AWS validation curve
p = (
    ROOT
    / "data"
    / "raw"
    / "aws"
    / "MEGHALAY_ISRO0041_2018-12-03_2019-01-03_Sep2026_177352.csv"
)
r = pd.read_csv(p)
temp = next(c for c in r.columns if c.startswith("AIR_TEMP"))
cols = [temp, "WIND_SPEED(m/s)", "ATMO_PRESSURE(hpa)", "HUMIDITY(%)", "RAIN_FALL(mm)"]
r["dt"] = pd.to_datetime(
    r["DATE(IST)"].astype(str) + " " + r["TIME(IST)"].astype(str), errors="coerce"
)
r = r.sort_values("dt").drop_duplicates("dt").set_index("dt")
for c in cols:
    r[c] = pd.to_numeric(r[c], errors="coerce")
d = r[cols].resample("30min").mean().interpolate(limit=2).ffill(limit=1).bfill(limit=1)
f = pd.DataFrame(index=d.index)
for c in [temp, "WIND_SPEED(m/s)", "ATMO_PRESSURE(hpa)", "HUMIDITY(%)"]:
    f[f"{c}_now"] = d[c]
    f[f"{c}_lag1"] = d[c].shift(1)
    f[f"{c}_lag2"] = d[c].shift(2)
    f[f"{c}_delta1"] = d[c].diff()
    f[f"{c}_mean3"] = d[c].rolling(3).mean()
f["hour"] = d.index.hour + d.index.minute / 60
f["sin_hour"] = np.sin(2 * np.pi * f.hour / 24)
f["cos_hour"] = np.cos(2 * np.pi * f.hour / 24)
targets = [temp, "HUMIDITY(%)", "ATMO_PRESSURE(hpa)", "WIND_SPEED(m/s)"]
z = f.join(d[targets].shift(-1)).dropna()
X = z[f.columns]
Y = z[targets]
s = int(len(z) * 0.8)
ns = [10, 25, 50, 100, 200, 400]
vals = []
for n in ns:
    m = ExtraTreesRegressor(
        n_estimators=n, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1
    )
    m.fit(X.iloc[:s], Y.iloc[:s])
    vals.append(
        float(
            mean_absolute_error(
                Y.iloc[s:], m.predict(X.iloc[s:]), multioutput="uniform_average"
            )
        )
    )
plt.figure(figsize=(8, 5))
plt.plot(ns, vals, marker="o")
plt.xlabel("Number of trees")
plt.ylabel("Mean validation MAE")
plt.title("AWS Shillong — Validation Error vs Number of Trees")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "aws_validation_curve.png", dpi=220)
plt.close()
(OUT / "aws_validation_curve.json").write_text(
    json.dumps(
        {
            "station": "ISRO0041_15F029(SHILLONG)",
            "trees": ns,
            "mean_validation_mae": vals,
        },
        indent=2,
    )
)
# GFS held-out evidence
m = json.loads((ROOT / "data" / "demo_gfs" / "metrics.json").read_text())
labels = ["Model MAE", "Persistence MAE"]
vals = [m["mae"], m["persistence_mae"]]
plt.figure(figsize=(7, 5))
plt.bar(labels, vals)
plt.ylabel("Normalized radar-field MAE")
plt.title("Held-out GFS-conditioned model vs persistence")
plt.tight_layout()
plt.savefig(OUT / "gfs_heldout_evidence.png", dpi=220)
plt.close()
(OUT / "gfs_heldout_evidence.json").write_text(json.dumps(m, indent=2))
(OUT / "README.md").write_text(
    "# PPT model evidence\n\n- aws_validation_curve.png: actual Shillong AWS ExtraTrees validation MAE versus tree count.\n- gfs_heldout_evidence.png: actual held-out GFS-conditioned model MAE versus persistence baseline.\n\nThe existing neural checkpoints do not contain epoch-by-epoch loss history, so genuine historical training curves cannot be reconstructed. Do not fabricate them. The neural training scripts should save train_loss/val_loss in future runs.\n"
)
print("PPT assets created")
