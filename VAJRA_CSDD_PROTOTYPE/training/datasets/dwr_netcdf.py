"""Prepare MOSDAC Cherrapunji DWR scans into day/event sequences."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import re
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

FILENAME_RE = re.compile(r"_(\d{2}[A-Z]{3}2026)_(\d{6})_L2B_STD\.nc$")


def parse_timestamp(path):
    m = FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse timestamp: {path.name}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%d%b%Y %H%M%S")


def _first_sweep(ds):
    elevation = np.asarray(ds["elevation"].values)
    split = np.where(np.diff(elevation) > 0.5)[0]
    end = int(split[0] + 1) if len(split) else elevation.shape[0] // 2
    return slice(0, end)


def polar_to_cartesian(dbz, azimuth, ranges, size=256, extent_km=300.0):
    az = np.deg2rad(azimuth.astype(np.float64))
    rr = ranges.astype(np.float64) / 1000.0
    x = rr[None, :] * np.sin(az[:, None])
    y = rr[None, :] * np.cos(az[:, None])
    points = np.column_stack((x.ravel(), y.ravel()))
    values = dbz.ravel()
    valid = np.isfinite(values)
    points, values = points[valid], values[valid]
    axis = np.linspace(-extent_km, extent_km, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    tree = cKDTree(points)
    dist, idx = tree.query(np.column_stack((xx.ravel(), yy.ravel())), k=1)
    out = values[idx].reshape(size, size).astype(np.float32)
    out[dist.reshape(size, size) > 2.0] = np.nan
    return out


def read_file(path, size=256, extent_km=300.0):
    with xr.open_dataset(path, decode_times=False) as ds:
        sweep = _first_sweep(ds)
        dbz = np.asarray(ds["DBZ"].values[sweep, :], dtype=np.float32)
        azimuth = np.asarray(ds["azimuth"].values[sweep], dtype=np.float32)
        ranges = np.asarray(ds["range"].values, dtype=np.float32)
    frame = polar_to_cartesian(dbz, azimuth, ranges, size, extent_km)
    frame = (frame + 10.0) / 80.0
    return np.clip(np.nan_to_num(frame, nan=0.0), 0.0, 1.0).astype(np.float32)
def split_into_sequences(files, max_gap_minutes=60.0, duplicate_gap_minutes=5.0, min_frames=12):
    files = sorted(files, key=parse_timestamp)
    sequences, current = [], []
    for path in files:
        if not current:
            current = [path]
            continue
        gap = (parse_timestamp(path) - parse_timestamp(current[-1])).total_seconds() / 60.0
        if gap < duplicate_gap_minutes:
            current[-1] = path
        elif gap <= max_gap_minutes:
            current.append(path)
        else:
            if len(current) >= min_frames:
                sequences.append(current)
            current = [path]
    if len(current) >= min_frames:
        sequences.append(current)
    return sequences


def prepare(input_dir="data/raw", output_dir="data/processed/events",
            size=256, extent_km=300.0):
    files = sorted(Path(input_dir).glob("*.nc"), key=parse_timestamp)
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {input_dir}")
    sequences = split_into_sequences(files)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for index, seq in enumerate(sequences):
        frames = np.stack([read_file(p, size, extent_km) for p in seq])
        start, end = parse_timestamp(seq[0]), parse_timestamp(seq[-1])
        gaps = [
            (parse_timestamp(b) - parse_timestamp(a)).total_seconds() / 60.0
            for a, b in zip(seq, seq[1:])
        ]
        name = f"segment_{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}.npy"
        np.save(outdir / name, frames)
        metadata.append({
            "file": name,
            "frames": int(frames.shape[0]),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "min_gap_minutes": min(gaps) if gaps else 0,
            "max_gap_minutes": max(gaps) if gaps else 0,
            "source_files": [p.name for p in seq],
        })
        print(f"[{index + 1:02d}/{len(sequences):02d}] {name} -> {frames.shape}")

    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Prepared {sum(x['frames'] for x in metadata)} frames in {len(metadata)} sequences.")
    print("Note: source scans are irregular; this baseline learns scan-to-scan evolution.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw")
    p.add_argument("--output", default="data/processed/events")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--extent-km", type=float, default=300.0)
    args = p.parse_args()
    prepare(args.input, args.output, args.size, args.extent_km)
