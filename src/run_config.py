# scripts/run_config.py
from pathlib import Path

import pandas as pd
import torch

from src.config import PerceptualConfig
from src.ct_dose_net import CTDoseNet, DoseNetFeatures
from src.distances import FeatureDistance, PerceptualObjective, PiqDistance
from src.feature_nets import build_feature_net
from src.train import train_and_test
from src.model import UNet  # the lab U-Net shared with tissue_band_nps_loss


def build_objective(config: PerceptualConfig, dose_net_path: Path, radimagenet: Path | None,
                    block_scales: list[float] | None = None) -> PerceptualObjective:
    if config.feature_net == "none":
        return PerceptualObjective(None, 0.0)
    if config.feature_net in ("lpips_vgg", "dists"):
        return PerceptualObjective(PiqDistance(config.feature_net), config.loss_weight, config.crop_size)
    if config.feature_net == "ct_dose_net":
        network = CTDoseNet()
        network.load_state_dict(torch.load(dose_net_path, map_location="cpu"))
        net = DoseNetFeatures(network)
    else:
        net = build_feature_net(config.feature_net, radimagenet)
    return PerceptualObjective(FeatureDistance(net, config.matching, block_scales), config.loss_weight, config.crop_size)


def run(config: PerceptualConfig, dose_net_path: Path, radimagenet: Path | None = None,
        block_scales: list[float] | None = None) -> pd.DataFrame:
    device = torch.device("cuda")
    objective = build_objective(config, dose_net_path, radimagenet, block_scales).to(device)
    return train_and_test(config, UNet().to(device), objective, device)