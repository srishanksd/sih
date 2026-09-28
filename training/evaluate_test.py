import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from training.models.convlstm import ConvLSTMNowcaster
from training.datasets.sequence import RadarSequenceDataset

ck = torch.load("training/checkpoints/best.pt", map_location="cuda")
model = ConvLSTMNowcaster(1, (32, 64), 1, 3).cuda()
model.load_state_dict(ck["model"])
model.eval()
ds = RadarSequenceDataset(["data/processed/cherrapunji_20260926.npy"], 8, 4, 1)
errs, pers = [], []
with torch.no_grad():
    for i in range(len(ds)):
        x, y = ds[i]
        x, y = x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda()
        pred = model(x, 4)
        errs.append(torch.mean(torch.abs(pred-y)).item())
        pers.append(torch.mean(torch.abs(x[:, -1:]-y[:,0:1])).item())
print("test_windows", len(ds))
print("model_MAE", float(np.mean(errs)))
print("persistence_first_step_MAE", float(np.mean(pers)))
print("improvement_pct", float(100*(1-np.mean(errs)/np.mean(pers))))
