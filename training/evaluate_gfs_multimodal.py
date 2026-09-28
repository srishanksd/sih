"""Evaluate the GFS-conditioned ConvLSTM on chronological held-out events."""
from pathlib import Path
import sys, json, numpy as np, torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.datasets.multimodal_sequence import MultimodalSequenceDataset
from training.models.convlstm import ConvLSTMNowcaster

DATA = ROOT / "data" / "processed" / "multimodal" / "gfs_features"
CK = ROOT / "training" / "checkpoints_gfs_multimodal" / "best.pt"
OUT = ROOT / "data" / "demo_gfs"
OUT.mkdir(exist_ok=True)

def split(files):
    files = sorted(files)
    n = len(files)
    a = max(1, int(n * .70))
    b = max(1, int(n * .15))
    if a + b >= n:
        a, b = n - 2, 1
    return files[:a], files[a:a+b], files[a+b:]

def main():
    files = sorted(DATA.glob("segment_*.npy"))
    _, _, te = split(files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultimodalSequenceDataset(te, 8, 4, 1)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    ck = torch.load(CK, map_location=device, weights_only=False)
    model = ConvLSTMNowcaster(23, (32, 64), 1, 3).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    maes, pmaes, first = [], [], None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, 4).clamp(0, 1)
            maes.append(float(torch.mean(torch.abs(pred - y)).cpu()))
            pmaes.append(float(torch.mean(torch.abs(x[:, -1:, :1] - y)).cpu()))
            if first is None:
                first = (x.cpu().numpy(), pred.cpu().numpy(), y.cpu().numpy())
    result = {
        "checkpoint_epoch": ck.get("epoch"),
        "test_windows": len(ds),
        "mae": float(np.mean(maes)),
        "persistence_mae": float(np.mean(pmaes)),
        "beats_persistence": bool(np.mean(maes) < np.mean(pmaes)),
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    # Dashboard/replay fields must be 2-D radar arrays, not full model tensors.
    observed = first[0][0, -1, 0]          # [H,W], latest input radar frame
    forecast = first[1][0, :, 0]            # [lead,H,W]
    actual = first[2][0, :, 0]              # [lead,H,W]
    observed_sequence = first[0][0, :, 0]   # [history,H,W]
    np.save(OUT / "observed.npy", observed)
    np.save(OUT / "observed_sequence.npy", observed_sequence)
    np.save(OUT / "forecast.npy", forecast)
    np.save(OUT / "actual_future.npy", actual)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
