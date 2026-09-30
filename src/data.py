from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def patient_id(path: Path) -> str:
    return path.stem.split("_")[0]


def assert_no_patient_leakage(data_root: Path) -> dict[str, int]:
    patients = {split: {patient_id(p) for p in (data_root / split).glob("*.npz")} for split in ("train", "val", "test")}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = patients[first] & patients[second]
        assert not shared, f"patients in both {first} and {second}: {sorted(shared)[:5]}"
    return {split: len(ids) for split, ids in patients.items()}


def load_slice(path: Path, key: str) -> torch.Tensor:
    with np.load(path) as sample:
        return torch.from_numpy(sample[key].astype(np.float32)).unsqueeze(0)


class I2IPairs(Dataset):
    def __init__(self, split_dir: Path, input_key: str, target_key: str) -> None:
        self.files = sorted(split_dir.glob("*.npz"))
        self.input_key = input_key
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path = self.files[index]
        return load_slice(path, self.input_key), load_slice(path, self.target_key), path.stem