# src/val_diagnostics.py
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import I2IPairs, patient_id
from src.model import UNet


@torch.no_grad()
def per_slice_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Per validation slice: L1 as in `validation_l1` (sum over pixels), mean signed error, and the L1 of a copy
    of the model whose BatchNorm layers use batch statistics. The model itself is left untouched."""
    model.eval()
    probe = copy.deepcopy(model).train()
    rows = []
    for noisy, clean, stems in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        prediction, batch_prediction = model(noisy), probe(noisy)
        for index, stem in enumerate(stems):
            error = prediction[index] - clean[index]
            rows.append({"slice_id": stem, "patient_id": patient_id(Path(stem)), "l1": error.abs().sum().item(),
                         "signed_error": error.mean().item(), "pixels": error.numel(),
                         "l1_batch_statistics": (batch_prediction[index] - clean[index]).abs().sum().item(),
                         "prediction_mean": prediction[index].mean().item(), "target_mean": clean[index].mean().item()})
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> dict[str, float]:
    ordered = frame["l1"].sort_values(ascending=False)
    return {"val_l1": frame["l1"].mean(), "val_l1_batch_statistics": frame["l1_batch_statistics"].mean(),
            "median_slice_l1": frame["l1"].median(), "max_slice_l1": frame["l1"].max(),
            "share_of_worst_10_slices": ordered.head(10).sum() / ordered.sum(),
            "mean_signed_error": frame["signed_error"].mean(),
            "share_of_l1_from_offset": (frame["signed_error"].abs() * frame["pixels"]).sum() / frame["l1"].sum()}


def best_checkpoint_report(data_root: Path, run_dirs: list[Path], output_dir: Path, device: torch.device) -> pd.DataFrame:
    """Validation diagnostics of saved best checkpoints (validation split only). Works on CPU."""
    loader = DataLoader(I2IPairs(data_root / "val", "noisy_clipped_100k", "gt_clipped"), batch_size=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in run_dirs:
        model = UNet().to(device)
        model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
        frame = per_slice_validation(model, loader, device)
        frame.to_csv(output_dir / f"{run_dir.name}_best_val_per_slice.csv", index=False)
        logged = json.loads((run_dir / "run_summary.json").read_text())["best_val_l1"]
        rows.append({"run": run_dir.name, "logged_best_val_l1": logged, **summarise(frame)})
        print(rows[-1])
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "best_checkpoint_summary.csv", index=False)
    return report


if __name__ == "__main__":      # python -m src.val_diagnostics DATA_ROOT train_C10_v2 train_C0 ...
    root = Path(__file__).resolve().parent.parent / "results"
    best_checkpoint_report(Path(sys.argv[1]), [root / "seed42" / name for name in sys.argv[2:]], root / "val_diagnostics",
                           torch.device("cuda" if torch.cuda.is_available() else "cpu"))
