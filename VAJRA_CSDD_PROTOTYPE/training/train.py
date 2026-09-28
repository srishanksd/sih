"""Train and evaluate the radar ConvLSTM baseline on chronological DWR events."""
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from training.datasets.sequence import RadarSequenceDataset
from training.losses.forecast import WeightedForecastLoss
from training.models.convlstm import ConvLSTMNowcaster


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_events(files, train_fraction=0.70, val_fraction=0.15):
    files = sorted(files)
    n = len(files)
    if n < 3:
        raise RuntimeError("Need at least 3 event sequences for train/validation/test.")
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
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


def main(config_path="training/configs/base.yaml"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]

    files = sorted(Path(data_cfg["root"]).glob("segment_*.npy"))
    train_files, val_files, test_files = split_events(
        files, data_cfg["train_fraction"], data_cfg["val_fraction"]
    )
    print(f"device={device}")
    print(f"events: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    train_ds = RadarSequenceDataset(train_files, data_cfg["sequence_length"],
                                    data_cfg["forecast_steps"], data_cfg["stride"])
    val_ds = RadarSequenceDataset(val_files, data_cfg["sequence_length"],
                                  data_cfg["forecast_steps"], data_cfg["stride"])
    test_ds = RadarSequenceDataset(test_files, data_cfg["sequence_length"],
                                   data_cfg["forecast_steps"], data_cfg["stride"])
    print(f"windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if not len(train_ds) or not len(val_ds) or not len(test_ds):
        raise RuntimeError("One split has zero temporal windows. Need more usable event data.")

    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                              shuffle=True, num_workers=data_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=data_cfg["batch_size"],
                            shuffle=False, num_workers=data_cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=data_cfg["batch_size"],
                             shuffle=False, num_workers=data_cfg["num_workers"])

    model = ConvLSTMNowcaster(
        model_cfg["in_channels"], tuple(model_cfg["hidden_channels"]),
        model_cfg["out_channels"], model_cfg["kernel_size"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    loss_fn = WeightedForecastLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, scaler,
                               device, data_cfg["forecast_steps"], True)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer, scaler,
                             device, data_cfg["forecast_steps"], False)
        print(f"epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f}")
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "config": cfg, "val_loss": val_loss
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, ckpt_dir / "best.pt")

    test_loss = run_epoch(model, test_loader, loss_fn, optimizer, scaler,
                          device, data_cfg["forecast_steps"], False)
    print(f"test_loss={test_loss:.5f}")
    print("Training complete. Best checkpoint: training/checkpoints/best.pt")


if __name__ == "__main__":
    main()
