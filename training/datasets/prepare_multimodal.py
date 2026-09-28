"""Build paired DWR + INSAT-3DR CTP sequences for Cherrapunji.

Radar remains the prediction target; INSAT CTP/CTT are auxiliary channels.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import re
import numpy as np
import h5py
from scipy.spatial import cKDTree
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.datasets.dwr_netcdf import parse_timestamp, read_file

ROOT = Path(__file__).resolve().parents[2]
DWR_DIR = ROOT / "data" / "raw" / "dwr"
INSAT_DIR = ROOT / "data" / "raw" / "insat_ctp"
OUT_DIR = ROOT / "data" / "processed" / "multimodal"
INSAT_RE = re.compile(r"3RIMG_(\d{2}[A-Z]{3}2026)_(\d{4})_L2B_CTP_V\d+R\d+\.h5$")
RADAR_LAT, RADAR_LON = 25.2680, 91.7332

def insat_timestamp(path: Path) -> datetime:
    m = INSAT_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse INSAT timestamp: {path.name}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%d%b%Y %H%M")

def _read_ctp(path: Path):
    with h5py.File(path, "r") as f:
        lat = f["Latitude"][...].astype(np.float32)
        lon = f["Longitude"][...].astype(np.float32)
        ctp = f["CTP"][0].astype(np.float32)
        ctt = f["CTT"][0].astype(np.float32)
        lat_a = dict(f["Latitude"].attrs)
        lon_a = dict(f["Longitude"].attrs)
        ctp_a = dict(f["CTP"].attrs)
        ctt_a = dict(f["CTT"].attrs)
    for arr, attrs in ((lat, lat_a), (lon, lon_a)):
        fill = float(np.asarray(attrs["_FillValue"]).ravel()[0])
        arr[arr == fill] = np.nan
        arr[:] = arr * float(np.asarray(attrs["scale_factor"]).ravel()[0])
        arr[:] = arr + float(np.asarray(attrs["add_offset"]).ravel()[0])
    for arr, attrs in ((ctp, ctp_a), (ctt, ctt_a)):
        fill = float(np.asarray(attrs["_FillValue"]).ravel()[0])
        arr[arr == fill] = np.nan
    return lat, lon, ctp, ctt
def _radar_target(size=256, extent_km=300.0):
    axis = np.linspace(-extent_km, extent_km, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    lat = RADAR_LAT + yy / 111.32
    lon = RADAR_LON + xx / (111.32 * np.cos(np.deg2rad(RADAR_LAT)))
    return lat, lon

def _map_insat(path: Path, target_lat, target_lon):
    lat, lon, ctp, ctt = _read_ctp(path)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(ctp) & np.isfinite(ctt)
    if valid.sum() < 1000:
        raise ValueError(f"Too few valid INSAT pixels in {path.name}: {valid.sum()}")
    pts = np.column_stack((lat[valid], lon[valid]))
    target = np.column_stack((target_lat.ravel(), target_lon.ravel()))
    tree = cKDTree(pts)
    dist, idx = tree.query(target, k=1)
    # INSAT CTP grid spacing is about 0.46 degrees in this product; use a
    # 0.30-degree acceptance radius to avoid discarding most valid pixels.
    ok = dist <= 0.30
    ctp_v = ctp[valid][idx]
    ctt_v = ctt[valid][idx]
    ctp_n = np.clip((800.0 - ctp_v) / 700.0, 0.0, 1.0)
    ctt_n = np.clip((300.0 - ctt_v) / 120.0, 0.0, 1.0)
    ctp_n[~ok] = 0.0
    ctt_n[~ok] = 0.0
    mask = ok.astype(np.float32)
    return np.stack([ctp_n, ctt_n, mask], axis=0).reshape(
        3, target_lat.shape[0], target_lat.shape[1]
    ).astype(np.float32)

def _nearest_insat(ts, insat_files, max_minutes=20.0):
    best = min(insat_files, key=lambda p: abs((insat_timestamp(p) - ts).total_seconds()))
    delta = abs((insat_timestamp(best) - ts).total_seconds()) / 60.0
    return (best, delta) if delta <= max_minutes else (None, delta)

def prepare(size=256, extent_km=300.0, max_match_minutes=20.0):
    dwr_files = sorted(DWR_DIR.glob("*.nc"), key=parse_timestamp)
    insat_files = sorted(INSAT_DIR.glob("*.h5"), key=insat_timestamp)
    if not dwr_files or not insat_files:
        raise FileNotFoundError("Need DWR .nc and INSAT CTP .h5 files.")
    target_lat, target_lon = _radar_target(size, extent_km)
    pairs, unmatched = [], []
    for dwr in dwr_files:
        ts = parse_timestamp(dwr)
        insat, delta = _nearest_insat(ts, insat_files, max_match_minutes)
        if insat is None:
            unmatched.append((dwr.name, round(delta, 2)))
        else:
            pairs.append((dwr, insat, delta))
    print(f"DWR files={len(dwr_files)} INSAT files={len(insat_files)}")
    print(f"paired={len(pairs)} unmatched={len(unmatched)} max_match={max_match_minutes} min")
    if unmatched:
        print("first unmatched:", unmatched[:10])
    insat_cache = {}
    paired_dwr, paired_sat, paired_meta = [], [], []
    for i, (dwr, insat, delta) in enumerate(pairs, 1):
        if insat.name not in insat_cache:
            insat_cache[insat.name] = _map_insat(insat, target_lat, target_lon)
        radar = read_file(dwr, size, extent_km)
        paired_dwr.append(radar)
        paired_sat.append(insat_cache[insat.name])
        paired_meta.append({
            "dwr": dwr.name,
            "insat": insat.name,
            "dwr_time": parse_timestamp(dwr).isoformat(),
            "insat_time": insat_timestamp(insat).isoformat(),
            "match_minutes": round(delta, 3),
        })
        if i % 20 == 0:
            print(f"paired frame {i}/{len(pairs)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dwr_arr = np.stack(paired_dwr).astype(np.float32)
    sat_arr = np.stack(paired_sat).astype(np.float32)
    combined = np.concatenate([dwr_arr[:, None], sat_arr], axis=1)
    np.save(OUT_DIR / "paired.npy", combined)
    np.save(OUT_DIR / "radar_target.npy", dwr_arr)
    np.save(OUT_DIR / "insat_features.npy", sat_arr)

    seqs, current = [], []
    for idx, meta in enumerate(paired_meta):
        ts = datetime.fromisoformat(meta["dwr_time"])
        if not current:
            current = [idx]
            continue
        prev = datetime.fromisoformat(paired_meta[current[-1]]["dwr_time"])
        gap = (ts - prev).total_seconds() / 60.0
        if gap <= 60.0:
            current.append(idx)
        else:
            if len(current) >= 12:
                seqs.append(current)
            current = [idx]
    if len(current) >= 12:
        seqs.append(current)
    events = []
    for idxs in seqs:
        start = datetime.fromisoformat(paired_meta[idxs[0]]["dwr_time"])
        end = datetime.fromisoformat(paired_meta[idxs[-1]]["dwr_time"])
        name = f"segment_{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}.npy"
        np.save(OUT_DIR / name, combined[idxs])
        events.append({
            "file": name,
            "frames": len(idxs),
            "start": start.isoformat(),
            "end": end.isoformat(),
        })
    (OUT_DIR / "metadata.json").write_text(
        json.dumps({"pairs": paired_meta, "events": events}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved combined={combined.shape} events={len(events)}")
    print("Channels: 0=DWR, 1=CTP signal, 2=CTT signal, 3=INSAT valid mask")

if __name__ == "__main__":
    # Multimodal fusion is limited by INSAT's coarser native grid, so the
    # prototype uses a 128x128 radar grid for efficient local training.
    prepare(size=128)
