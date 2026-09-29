from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "aws" / "MEGHALAY_ISRO0041_2018-12-03_2019-01-03_Sep2026_177352.csv"
OUT = ROOT / "training" / "checkpoints" / "aws_forecaster.joblib"
METRICS = ROOT / "training" / "checkpoints" / "aws_forecaster_metrics.json"

raw = pd.read_csv(CSV)
temp = next(c for c in raw.columns if c.startswith("AIR_TEMP"))
wind = "WIND_SPEED(m/s)"
pressure = "ATMO_PRESSURE(hpa)"
humidity = "HUMIDITY(%)"
rain = "RAIN_FALL(mm)"
raw["dt"] = pd.to_datetime(raw["DATE(IST)"].astype(str) + " " + raw["TIME(IST)"].astype(str), dayfirst=False)
raw = raw.sort_values("dt").drop_duplicates("dt").set_index("dt")

cols = [temp, wind, pressure, humidity, rain]
for c in cols:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

# Keep the native half-hour timestamp grid; interpolate only short observation gaps.
df = raw[cols].resample("30min").mean()
df = df.interpolate(limit=2).ffill(limit=1).bfill(limit=1)

features = pd.DataFrame(index=df.index)
for c in [temp, wind, pressure, humidity]:
    features[f"{c}_now"] = df[c]
    features[f"{c}_lag1"] = df[c].shift(1)
    features[f"{c}_lag2"] = df[c].shift(2)
    features[f"{c}_delta1"] = df[c].diff(1)
    features[f"{c}_mean3"] = df[c].rolling(3).mean()
features["hour"] = df.index.hour + df.index.minute / 60.0
features["sin_hour"] = np.sin(2 * np.pi * features["hour"] / 24)
features["cos_hour"] = np.cos(2 * np.pi * features["hour"] / 24)

# Forecast the next 30-minute surface state; this is the highest-fidelity target
# supported by the irregular AWS sampling in this archive.
target_cols = [temp, humidity, pressure, wind]
y = df[target_cols].shift(-1)
data = features.join(y, how="inner").dropna()
X = data[features.columns]
Y = data[target_cols]

# Strict chronological split: no random shuffling and no future leakage.
split = int(len(data) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
Y_train, Y_test = Y.iloc[:split], Y.iloc[split:]

model = ExtraTreesRegressor(
    n_estimators=400, min_samples_leaf=2, max_features=0.8,
    random_state=42, n_jobs=-1
)
model.fit(X_train, Y_train)
pred = model.predict(X_test)

metrics = {}
for i, c in enumerate(target_cols):
    persistence = X_test[f"{c}_now"].to_numpy()
    metrics[c] = {
        "mae_model": float(mean_absolute_error(Y_test.iloc[:, i], pred[:, i])),
        "rmse_model": float(np.sqrt(mean_squared_error(Y_test.iloc[:, i], pred[:, i]))),
        "r2_model": float(r2_score(Y_test.iloc[:, i], pred[:, i])),
        "mae_persistence": float(mean_absolute_error(Y_test.iloc[:, i], persistence)),
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "features": list(X.columns), "targets": target_cols,
             "station": str(raw["@STATION_ID"].iloc[0])}, OUT)
summary = {"station": str(raw["@STATION_ID"].iloc[0]),
           "date_start": str(raw.index.min()), "date_end": str(raw.index.max()),
           "raw_rows": int(len(raw)), "resampled_rows": int(len(df)),
           "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
           "forecast_horizon": "30 minutes", "metrics": metrics}
METRICS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"saved model: {OUT}")
