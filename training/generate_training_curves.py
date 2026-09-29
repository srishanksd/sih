import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

ASSET = ROOT / "assets" / "training_curves"
ASSET.mkdir(parents=True, exist_ok=True)


def seed(seed=84):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


from training.datasets.multimodal_sequence import MultimodalSequenceDataset
from training.datasets.sequence import RadarSequenceDataset
from training.losses.forecast import WeightedForecastLoss
from training.models.convlstm import ConvLSTMNowcaster


def split(files):
    files = sorted(files)
    n = len(files)
    a = max(1, int(n * 0.70))
    b = max(1, int(n * 0.15))
    a, b = (n - 2, 1) if a + b >= n else (a, b)
    return files[:a], files[a : a + b], files[a + b :]


def ep(model, loader, lossfn, opt, scaler, device, train):
    model.train(train)
    total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                pred = model(x, 4)
                loss = lossfn(pred, y)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(opt)
                scaler.update()
            total += loss.item()
    return total / max(len(loader), 1)


def neural(name, files, mm, ch, epochs=5):
    seed()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = split(files)
    DS = MultimodalSequenceDataset if mm else RadarSequenceDataset
    a, b, c = DS(tr, 8, 4, 1), DS(va, 8, 4, 1), DS(te, 8, 4, 1)
    print(f"{name}: device={dev}, windows={len(a)}/{len(b)}/{len(c)}", flush=True)
    tl = DataLoader(a, batch_size=2, shuffle=True, num_workers=0)
    vl = DataLoader(b, batch_size=2, shuffle=False, num_workers=0)
    el = DataLoader(c, batch_size=2, shuffle=False, num_workers=0)
    m = ConvLSTMNowcaster(ch, (32, 64), 1, 3).to(dev)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    lf = WeightedForecastLoss()
    sc = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    trh = []
    vh = []
    for e in range(1, epochs + 1):
        x = ep(m, tl, lf, o, sc, dev, True)
        y = ep(m, vl, lf, o, sc, dev, False)
        trh.append(x)
        vh.append(y)
        print(f"{name} epoch={e} train={x:.6f} val={y:.6f}", flush=True)
    test = ep(m, el, lf, o, sc, dev, False)
    out = {
        "model": name,
        "epochs": epochs,
        "train_loss": trh,
        "val_loss": vh,
        "test_loss": test,
        "note": "Lightweight diagnostic retraining run for visualization; existing production checkpoint is separate.",
    }
    (ASSET / f"{name}.json").write_text(json.dumps(out, indent=2))
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), trh, label="Training loss")
    plt.plot(range(1, epochs + 1), vh, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Forecast loss")
    plt.title(name)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET / f"{name}.png", dpi=180)
    plt.close()


def aws():
    p = ROOT / "data/raw/aws/MEGHALAY_ISRO0041_2018-12-03_2019-01-03_Sep2026_177352.csv"
    r = pd.read_csv(p)
    temp = next(c for c in r.columns if c.startswith("AIR_TEMP"))
    cols = [
        temp,
        "WIND_SPEED(m/s)",
        "ATMO_PRESSURE(hpa)",
        "HUMIDITY(%)",
        "RAIN_FALL(mm)",
    ]
    r["dt"] = pd.to_datetime(
        r["DATE(IST)"].astype(str) + " " + r["TIME(IST)"].astype(str), errors="coerce"
    )
    r = r.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in cols:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    d = (
        r[cols]
        .resample("30min")
        .mean()
        .interpolate(limit=2)
        .ffill(limit=1)
        .bfill(limit=1)
    )
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
    t = [temp, "HUMIDITY(%)", "ATMO_PRESSURE(hpa)", "WIND_SPEED(m/s)"]
    z = f.join(d[t].shift(-1)).dropna()
    X = z[f.columns]
    Y = z[t]
    s = int(len(z) * 0.8)
    ns = [10, 25, 50, 100, 200, 400]
    vals = []
    for n in ns:
        m = ExtraTreesRegressor(
            n_estimators=n,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X.iloc[:s], Y.iloc[:s])
        vals.append(
            float(
                mean_absolute_error(
                    Y.iloc[s:], m.predict(X.iloc[s:]), multioutput="uniform_average"
                )
            )
        )
        print(f"AWS trees={n} MAE={vals[-1]:.6f}", flush=True)
    (ASSET / "aws_extratrees.json").write_text(
        json.dumps({"trees": ns, "mean_validation_mae": vals}, indent=2)
    )
    plt.figure(figsize=(8, 5))
    plt.plot(ns, vals, marker="o")
    plt.xlabel("Number of trees")
    plt.ylabel("Mean validation MAE")
    plt.title("AWS ExtraTrees — Validation Error vs Trees")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ASSET / "aws_extratrees.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    neural(
        "radar_convlstm",
        sorted((ROOT / "data/processed/events").glob("segment_*.npy")),
        False,
        1,
    )
    neural(
        "dwr_insat_convlstm",
        sorted((ROOT / "data/processed/multimodal").glob("segment_*.npy")),
        True,
        4,
    )
    neural(
        "dwr_insat_gfs_convlstm",
        sorted((ROOT / "data/processed/multimodal/gfs_features").glob("segment_*.npy")),
        True,
        23,
    )
    aws()
    (ASSET / "README.md").write_text(
        "# Training curves\n\nNeural figures are lightweight 5-epoch diagnostic retraining runs on the repository datasets. AWS is validation MAE versus tree count because ExtraTrees has no epoch history. These figures are for model-behavior visualization, not claims of operational skill.\n"
    )
    print("CURVES COMPLETE", flush=True)
