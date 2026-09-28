"""Convert a time-ordered radar array into the standard training format."""
from pathlib import Path
import numpy as np

def prepare_array(source, output, scale_min=0.0, scale_max=1.0):
    arr = np.load(source).astype(np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,H,W] or [T,1,H,W], got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=scale_max, neginf=scale_min)
    if scale_max > scale_min:
        arr = (arr - scale_min) / (scale_max - scale_min)
    arr = np.clip(arr, 0.0, 1.0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, arr.astype(np.float32))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--min", dest="minimum", type=float, default=0.0)
    parser.add_argument("--max", dest="maximum", type=float, default=1.0)
    args = parser.parse_args()
    prepare_array(args.source, args.output, args.minimum, args.maximum)