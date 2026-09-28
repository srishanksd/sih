"""Convert MOSDAC DWR L2B NetCDF files into fixed Cartesian DBZ frames."""
from pathlib import Path
import re
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

FILENAME_RE = re.compile(r"_(\d{2})(\d{2})(\d{2})_L2B")


def timestamp_from_name(path):
    match = FILENAME_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse HHMMSS from {path}")
    return ":".join(match.groups())


def _first_sweep(ds):
    elevation = np.asarray(ds["elevation"].values)
    # These files contain 360 rays per sweep; use the lowest elevation for
    # the first prototype because it is closest to the surface.
    split = np.where(np.diff(elevation) > 0.5)[0]
    end = int(split[0] + 1) if len(split) else elevation.shape[0] // 2
    return slice(0, end)


def polar_to_cartesian(dbz, azimuth, ranges, size=256, extent_km=300.0):
    az = np.deg2rad(azimuth.astype(np.float64))
    rr = ranges.astype(np.float64) / 1000.0
    # Radar convention: azimuth 0 = north, increasing clockwise.
    x = rr[None, :] * np.sin(az[:, None])
    y = rr[None, :] * np.cos(az[:, None])
    points = np.column_stack((x.ravel(), y.ravel()))
    values = dbz.ravel()
    valid = np.isfinite(values)
    points = points[valid]
    values = values[valid]
    axis = np.linspace(-extent_km, extent_km, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    tree = cKDTree(points)
    dist, idx = tree.query(np.column_stack((xx.ravel(), yy.ravel())), k=1)
    out = values[idx].reshape(size, size).astype(np.float32)
    out[dist.reshape(size, size) > 2.0] = np.nan
    return out


def normalize_dbz(dbz, minimum=-10.0, maximum=70.0):
    out = (dbz - minimum) / (maximum - minimum)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def read_file(path, size=256, extent_km=300.0):
    with xr.open_dataset(path, decode_times=False) as ds:
        sweep = _first_sweep(ds)
        dbz = np.asarray(ds["DBZ"].values[sweep, :], dtype=np.float32)
        azimuth = np.asarray(ds["azimuth"].values[sweep], dtype=np.float32)
        ranges = np.asarray(ds["range"].values, dtype=np.float32)
        frame = polar_to_cartesian(dbz, azimuth, ranges, size, extent_km)
        return normalize_dbz(frame), timestamp_from_name(path)


def build_event(input_dir, output, size=256, extent_km=300.0, max_gap_minutes=20.0):
    files = sorted(Path(input_dir).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {input_dir}")
    frames, times, kept = [], [], []
    for path in files:
        frame, hhmmss = read_file(path, size, extent_km)
        frames.append(frame)
        times.append(hhmmss)
        kept.append(path.name)
    # Keep every file for the prototype, but write metadata so the trainer can
    # reject non-uniform sequences. The source product can have irregular scan times.
    arr = np.stack(frames).astype(np.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, arr)
    output.with_suffix(".txt").write_text(
        "\\n".join(f"{name}\\t{t}" for name, t in zip(kept, times)), encoding="utf-8"
    )
    return arr.shape, times


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/radar")
    parser.add_argument("--output", default="data/processed/cherrapunji_20260926.npy")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--extent-km", type=float, default=300.0)
    args = parser.parse_args()
    shape, times = build_event(args.input, args.output, args.size, args.extent_km)
    print(f"Processed {shape[0]} radar scans -> {shape}")
    print(f"First: {times[0]} | Last: {times[-1]}")
