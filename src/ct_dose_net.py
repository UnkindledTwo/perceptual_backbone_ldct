# src/ct_dose_net.py
import torch
import torch.nn as nn

from src.feature_nets import FeatureNet


def conv_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class CTDoseNet(nn.Module):
    def __init__(self, classes: int = 6) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([conv_block(1, 32, 1), conv_block(32, 64, 2), conv_block(64, 128, 2)])
        self.head = nn.Sequential(conv_block(128, 256, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, classes))

    def block_outputs(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs, x = [], image
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.block_outputs(image)[-1])


class DoseNetFeatures(FeatureNet):
    def __init__(self, network: CTDoseNet, input_size: int = 224) -> None:
        super().__init__(input_size, ((0.0,) * 3, (1.0,) * 3))
        self.network = network

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        return self.network.block_outputs(image)