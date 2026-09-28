from pathlib import Path
import json


def split_files(root, train_fraction=0.70, val_fraction=0.15):
    files = sorted(Path(root).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy arrays found in {root}")
    n = len(files)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction)) if n >= 3 else 0
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def save_split_manifest(path, train, val, test):
    payload = {"train": [str(p) for p in train], "val": [str(p) for p in val], "test": [str(p) for p in test]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
