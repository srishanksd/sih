"""Temporal windows for preprocessed DWR sequences."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class RadarSequenceDataset(Dataset):
    def __init__(self, files, input_steps=8, forecast_steps=4, stride=1):
        self.files = [Path(f) for f in files]
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.samples = []
        window = input_steps + forecast_steps
        for path in self.files:
            arr = np.load(path, mmap_mode="r")
            for start in range(0, max(0, arr.shape[0] - window + 1), stride):
                self.samples.append((path, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, start = self.samples[index]
        arr = np.load(path, mmap_mode="r")
        window = np.asarray(
            arr[start:start + self.input_steps + self.forecast_steps],
            dtype=np.float32,
        )
        window = np.nan_to_num(window, nan=0.0, posinf=1.0, neginf=0.0)
        window = np.clip(window, 0.0, 1.0)[:, None, :, :]
        x = torch.from_numpy(window[:self.input_steps])
        y = torch.from_numpy(window[self.input_steps:])
        return x, y
