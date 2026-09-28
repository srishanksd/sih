"""Build GFS atmospheric features aligned to the existing DWR+INSAT pairs."""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import cfgrib
import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "gfs"
META = ROOT / "data" / "processed" / "multimodal" / "metadata.json"
OUT = ROOT / "data" / "processed" / "gfs"
OUT.mkdir(parents=True, exist_ok=True)

# [u10,v10,t2m,d2m,cape,cin] + [t,u,v] at 850/700/500/300 hPa.
FEATURE_NAMES = [
    "u10", "v10", "t2m", "d2m", "cape", "cin",
    "t850", "u850", "v850", "t700", "u700", "v700",
    "t500", "u500", "v500", "t300", "u300", "v300",
]

def norm(name, a):
    ranges = {
        "u10": (-30, 30), "v10": (-30, 30),
        "t2m": (240, 330), "d2m": (230, 315),
        "cape": (0, 4000), "cin": (-500, 0),
    }
    if name in ranges:
        lo, hi = ranges[name]
    elif name.startswith("t"):
        lo, hi = 220, 320
    else:
        lo, hi = -60, 60
    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip((a - lo) / (hi - lo), 0, 1)

def read_gfs(path):
    groups = cfgrib.open_datasets(str(path), indexpath="")
    surface = {}
    upper = None
    for ds in groups:
        if "isobaricInhPa" in ds.dims:
            upper = ds
        for name in ds.data_vars:
            if "isobaricInhPa" not in ds.dims:
                surface[name] = ds[name]
    if upper is None:
        raise RuntimeError(f"No pressure-level group in {path.name}")
    lat = groups[0].latitude.values
    lon = groups[0].longitude.values
    lat_mask = (lat >= 20) & (lat <= 30)
    lon_mask = (lon >= 85) & (lon <= 100)
    li = np.where(lat_mask)[0]
    lj = np.where(lon_mask)[0]
    lat_slice = slice(li.min(), li.max() + 1)
    lon_slice = slice(lj.min(), lj.max() + 1)
    fields = []
    for name in FEATURE_NAMES:
        if name in ("u10", "v10", "t2m", "d2m", "cape", "cin"):
            src = surface[name].values[lat_slice, lon_slice]
        else:
            level = int(name[1:])
            base = name[0]
            src = upper[base].sel(isobaricInhPa=level).values[lat_slice, lon_slice]
        x = norm(name, src)
        zy = 128 / x.shape[0]
        zx = 128 / x.shape[1]
        fields.append(zoom(x, (zy, zx), order=1).astype(np.float32))
    valid = groups[0].valid_time.values
    return np.stack(fields), np.datetime64(valid)

def file_valid_time(path):
    m = re.search(r"gfs\.(\d{8})\.t00z\.f(\d{3})\.grib2$", path.name)
    if not m:
        return None
    base = datetime.strptime(m.group(1), "%Y%m%d")
    return np.datetime64(base + timedelta(hours=int(m.group(2))))

files = sorted(RAW.glob("gfs.202609*.grib2"))
valid_files = [(p, file_valid_time(p)) for p in files]
valid_files = [(p, t) for p, t in valid_files if t is not None]
valid_files.sort(key=lambda z: z[1])
print(f"GFS files: {len(valid_files)}")

meta = json.loads(META.read_text())
pairs = meta["pairs"]
features, used_times, used_dwr = [], [], []
cache = {}
for pair in pairs:
    dt = np.datetime64(datetime.fromisoformat(pair["dwr_time"]))
    candidates = [(p, t) for p, t in valid_files if t <= dt]
    if not candidates:
        continue
    p, vt = candidates[-1]
    key = str(p)
    if key not in cache:
        cache[key] = read_gfs(p)
        print("parsed", p.name, cache[key][0].shape, "valid", cache[key][1])
    arr, actual_vt = cache[key]
    features.append(arr)
    used_times.append(str(actual_vt))
    used_dwr.append(pair["dwr_time"])

if not features:
    raise RuntimeError("No DWR pairs could be matched to GFS files.")
X = np.stack(features).astype(np.float32)
np.save(OUT / "features.npy", X)
(OUT / "dwr_times.json").write_text(json.dumps(used_dwr, indent=2))
(OUT / "gfs_valid_times.json").write_text(json.dumps(used_times, indent=2))
(OUT / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2))
print("saved", X.shape, "to", OUT)
