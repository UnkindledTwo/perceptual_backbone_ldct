from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PerceptualConfig
from src.data import I2IPairs, patient_id
from src.metrics import save_per_slice_metrics, slice_metrics


def validation_l1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for noisy, clean, _ in loader:
            total += F.l1_loss(model(noisy.to(device)), clean.to(device), reduction="sum").item()
    return total / len(loader.dataset)


def train_and_test(config: PerceptualConfig, model: nn.Module, criterion: nn.Module,
                   device: torch.device) -> pd.DataFrame:
    splits = {s: I2IPairs(config.data_root / s, config.input_key, config.target_key) for s in ("train", "val", "test")}
    train = DataLoader(splits["train"], batch_size=config.batch_size, shuffle=True, num_workers=4)
    val = DataLoader(splits["val"], batch_size=config.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    scaler = torch.amp.GradScaler(device.type, enabled=config.use_amp)
    best, checkpoint = float("inf"), config.run_dir / "best.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for _ in tqdm(range(config.epochs), desc=config.run_name):
        model.train()
        for step, (noisy, clean, _) in enumerate(train):
            noisy, clean = noisy.to(device), clean.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.use_amp):
                prediction = model(noisy)
            loss = criterion(prediction.float(), clean) / config.grad_accumulation
            scaler.scale(loss).backward()
            if (step + 1) % config.grad_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        score = validation_l1(model, val, device)
        scheduler.step(score)
        if score < best:
            best = score
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    rows = []
    with torch.no_grad():
        for noisy, clean, stem in splits["test"]:
            prediction = model(noisy[None].to(device)).clamp(0, 1)[0, 0].cpu().numpy()
            rows.append({"slice_id": stem, "patient_id": patient_id(Path(stem)),
                         **slice_metrics(prediction, clean[0].numpy())})
    return save_per_slice_metrics(rows, config.run_dir / "metrics.csv")

def only_test(config: PerceptualConfig, model: nn.Module, checkpoint: Path,
                   device: torch.device) -> pd.DataFrame:
    splits = {s: I2IPairs(config.data_root / s, config.input_key, config.target_key) for s in ("train", "val", "test")}
    best, checkpoint = float("inf"), checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    rows = []
    with torch.no_grad():
        for noisy, clean, stem in splits["test"]:
            prediction = model(noisy[None].to(device)).clamp(0, 1)[0, 0].cpu().numpy()
            rows.append({"slice_id": stem, "patient_id": patient_id(Path(stem)),
                         **slice_metrics(prediction, clean[0].numpy())})
    return save_per_slice_metrics(rows, config.run_dir / "metrics_test.csv")
