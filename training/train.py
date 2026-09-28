from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.sequence import RadarSequenceDataset
from losses.forecast import WeightedForecastLoss
from models.convlstm import ConvLSTMNowcaster
from utils.data_split import save_split_manifest, split_files


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, loss_fn, device, future_steps):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, future_steps)
            total += loss_fn(pred, y).item()
            count += 1
    return total / max(count, 1)


def main(config_path="training/configs/base.yaml"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    train_files, val_files, test_files = split_files(data_cfg["root"], data_cfg["train_fraction"], data_cfg["val_fraction"])
    save_split_manifest("data/splits/radar_split.json", train_files, val_files, test_files)
    train_ds = RadarSequenceDataset(train_files, data_cfg["sequence_length"], data_cfg["forecast_steps"], data_cfg["stride"])
    val_ds = RadarSequenceDataset(val_files, data_cfg["sequence_length"], data_cfg["forecast_steps"], data_cfg["stride"])
    if not len(train_ds):
        raise RuntimeError("No training windows found. Add preprocessed .npy radar sequences to data/processed first.")
    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"]) if len(val_ds) else None
    model = ConvLSTMNowcaster(model_cfg["in_channels"], tuple(model_cfg["hidden_channels"]), model_cfg["out_channels"], model_cfg["kernel_size"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    loss_fn = WeightedForecastLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"] and device.type == "cuda")
    best_val = float("inf")
    Path(train_cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=train_cfg["amp"] and device.type == "cuda"):
                pred = model(x, data_cfg["forecast_steps"])
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        train_loss = running / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn, device, data_cfg["forecast_steps"]) if val_loader else train_loss
        print(f"epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f} device={device}")
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "val_loss": val_loss}
        torch.save(checkpoint, Path(train_cfg["checkpoint_dir"]) / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, Path(train_cfg["checkpoint_dir"]) / "best.pt")


if __name__ == "__main__":
    main()
