from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, resnet50, vgg19

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_STATS = ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))


class FeatureNet(nn.Module):
    def __init__(self, input_size: int, stats: tuple[tuple[float, ...], tuple[float, ...]]) -> None:
        super().__init__()
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(stats[0]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(stats[1]).view(1, 3, 1, 1))

    def prepare(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=self.input_size, mode="bilinear", align_corners=False)
        return (image.expand(-1, 3, -1, -1) - self.mean) / self.std

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError


class VGGFeatures(FeatureNet):
    def __init__(self, layer_ends: tuple[int, ...] = (13,)) -> None:
        super().__init__(224, IMAGENET)
        self.body = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: max(layer_ends) + 1]
        self.layer_ends = layer_ends

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs, x = [], self.prepare(image)
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in self.layer_ends:
                outputs.append(x)
        return outputs


class ResNetFeatures(FeatureNet):
    def __init__(self, weights_path: Path) -> None:
        super().__init__(224, IMAGENET)  # TODO: RadImageNet normalization from the weights source
        self.body = resnet50(weights=None)
        self.body.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        b = self.body
        x = b.maxpool(b.relu(b.bn1(b.conv1(self.prepare(image)))))
        f1 = b.layer1(x)
        f2 = b.layer2(f1)
        return [f1, f2, b.layer3(f2)]


class DinoFeatures(FeatureNet):
    def __init__(self, size: str = "s", blocks: tuple[int, ...] = (3, 6, 9)) -> None:
        super().__init__(224, IMAGENET)
        self.body = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{size}14")
        self.blocks = blocks

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        return list(self.body.get_intermediate_layers(self.prepare(image), n=list(self.blocks), reshape=True))


class HFViTFeatures(FeatureNet):
    def __init__(self, vision_model: nn.Module, input_size: int, patch: int,
                 stats: tuple[tuple[float, ...], tuple[float, ...]], layers: tuple[int, ...], has_cls: bool) -> None:
        super().__init__(input_size, stats)
        self.body = vision_model
        self.grid = input_size // patch
        self.layers = layers
        self.has_cls = has_cls

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.body(pixel_values=self.prepare(image), output_hidden_states=True).hidden_states
        maps = []
        for layer in self.layers:
            tokens = hidden[layer][:, 1:] if self.has_cls else hidden[layer]
            maps.append(tokens.transpose(1, 2).reshape(tokens.shape[0], -1, self.grid, self.grid))
        return maps


def build_feature_net(name: str, radimagenet_weights: Path | None = None) -> FeatureNet:
    if name == "vgg19":
        return VGGFeatures()
    if name == "radimagenet":
        return ResNetFeatures(radimagenet_weights)
    if name in ("dinov2_s", "dinov2_b", "dinov2_l"):
        return DinoFeatures(name[-1], (6, 12, 18) if name == "dinov2_l" else (3, 6, 9))
    if name == "medsiglip":
        from transformers import AutoModel
        vision = AutoModel.from_pretrained("google/medsiglip-448").vision_model
        return HFViTFeatures(vision, 448, 14, ((0.5,) * 3, (0.5,) * 3), (7, 14, 20), has_cls=False)
    if name == "clip_b16":
        from transformers import CLIPVisionModel
        vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        return HFViTFeatures(vision, 224, 16, CLIP_STATS, (3, 6, 9), has_cls=True)
    raise ValueError(f"unknown feature network {name}")