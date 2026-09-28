"""Train ConvLSTM using DWR + INSAT CTP/CTT as multimodal inputs."""
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.multimodal_sequence import MultimodalSequenceDataset
from training.losses.forecast import WeightedForecastLoss
from training.models.convlstm import ConvLSTMNowcaster

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multimodal"
CKPT_DIR = ROOT / "training" / "checkpoints_multimodal"

def seed_everything(seed=84):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_events(files):
    files = sorted(files)
    n = len(files)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]

def run_epoch(model, loader, loss_fn, optimizer, scaler, device, future_steps, train):
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                pred = model(x, future_steps)
                loss = loss_fn(pred, y)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total += loss.item()
    return total / max(len(loader), 1)

def main(epochs=20):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(DATA_ROOT.glob("segment_*.npy"))
    train_files, val_files, test_files = split_events(files)
    print(f"device={device}")
    print(f"events: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    train_ds = MultimodalSequenceDataset(train_files, 8, 4, 1)
    val_ds = MultimodalSequenceDataset(val_files, 8, 4, 1)
    test_ds = MultimodalSequenceDataset(test_files, 8, 4, 1)
    print(f"windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if not all((len(train_ds), len(val_ds), len(test_ds))):
        raise RuntimeError("One split has zero windows.")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    model = ConvLSTMNowcaster(4, (32, 64), 1, 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = WeightedForecastLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, 4, True)
        va = run_epoch(model, val_loader, loss_fn, optimizer, scaler, device, 4, False)
        print(f"epoch={epoch:03d} train={tr:.5f} val={va:.5f}")
        checkpoint = {"epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "val_loss": va,
                      "in_channels": 4, "future_steps": 4}
        torch.save(checkpoint, CKPT_DIR / "last.pt")
        if va < best_val:
            best_val = va
            torch.save(checkpoint, CKPT_DIR / "best.pt")

    test = run_epoch(model, test_loader, loss_fn, optimizer, scaler, device, 4, False)
    print(f"test_loss={test:.5f}")
    print(f"best_checkpoint={CKPT_DIR / 'best.pt'}")

if __name__ == "__main__":
    main()
