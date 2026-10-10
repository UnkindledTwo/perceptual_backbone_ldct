# Paper Plan: Which Feature Network Makes a Good Perceptual Loss for LDCT?

[README](README.md) (research questions, contributions, figure and table list)
· [novelty.md](novelty.md) (related work and references)

**Paper type:** benchmark (with one proposed feature network)
**Track:** standard
**Assignee:** Arda Şahin ([@UnkindledTwo](https://github.com/UnkindledTwo))
**GitHub issue:** [#261](https://github.com/itu-biai/dlct_docs/issues/261)
until a dedicated ISBI issue exists (TODO)
**Dates:** results freeze Sun Oct 18, first draft to the team Fri Oct 23,
official ISBI deadline Mon Oct 26, planned submission Mon Nov 2. Fallback
without an extension: submit the first draft on Mon Oct 26.

## How to Use This Plan

Work from top to bottom. Each step lists what it produces for the paper, the
tasks, the formula, the code and the table to fill. Fill result cells in
place as runs finish, and keep the per-slice CSVs; the Final Tables are
generated from them. Post each week's summary as a Markdown comment on the
GitHub issue.

---

## Protocol

The fixed setup every step uses. Changing anything here after Week 1 means
re-running every earlier row.

- **Data:** `wiki/datasets/dataset_biailab_I2I_2025v1`, 512 × 512, values in
  [0, 1]. Training input `noisy_clipped_100k`, target `gt_clipped`. The
  screen in Step 1.4 also uses `noisy_clipped_10k`, `_50k`, `_500k` and
  `_1000k`.
- **Split:** the patient-level split shipped with the dataset (899 train, 182
  val, 220 test slices), checked in Step 1.1. The screen and the $\lambda$
  calibration use train or val only; the test split is touched once per run.
- **Backbone and budget:** the lab U-Net shared with `vlm_clean_prior_ssl_ldct`
  and `tissue_band_nps_loss`: AdamW, learning rate $10^{-4}$, batch 8, 30
  epochs, `ReduceLROnPlateau(factor=0.5)`, AMP. The checkpoint with the
  lowest **validation L1** is kept for every configuration, so no loss picks
  its own selection rule.
- **Objective of every configuration:**

$$
\mathcal{L}_\phi(\hat{x}, x) = \lVert \hat{x} - x \rVert_1 + \lambda_\phi \, d_\phi\big(\mathrm{crop}(\hat{x}), \mathrm{crop}(x)\big)
$$

where $\hat{x} = f_\theta(x_{\mathrm{LD}})$ is the U-Net output, $x$ the
normal-dose target, $d_\phi$ the feature distance of feature network $\phi$
(pointwise or statistics, Step 1.3), $\mathrm{crop}$ one random 224 × 224
crop shared by prediction and target and resized to the network's input size,
and $\lambda_\phi$ set so the perceptual term has the same initial share of
the loss for every network (Step 2.1).

- **Compared configurations (11):**

| ID | Feature network | Family | Pretraining | Matching | Source |
|---|---|---|---|---|---|
| C0 | none (L1) | reference | - | - | - |
| C1 | VGG-19, relu3_2 | ImageNet CNN | ImageNet classification | pointwise | `viana2025perceptual` (their best layer) |
| C2 | LPIPS-VGG | calibrated CNN | ImageNet + human judgments | pointwise (learned weights) | `zhang2018lpips` |
| C3 | DISTS | texture-statistics IQA | ImageNet + human judgments | statistics (structure and texture) | `ding2020dists` |
| C4 | RadImageNet ResNet-50, layer1 to layer3 | medical CNN | RadImageNet classification | pointwise | `mei2022radimagenet` |
| C5 | DINOv2 ViT-S/14, blocks 3, 6, 9 | self-supervised ViT, small | LVD-142M, self-distillation | pointwise | `oquab2024dinov2` |
| C6 | DINOv2 ViT-B/14, blocks 3, 6, 9 | self-supervised ViT, base | LVD-142M, self-distillation | pointwise | `oquab2024dinov2` |
| C7 | MedSigLIP-448, layers 7, 14, 20 | medical VLM | medical image and text pairs | pointwise | `sellergren2025medgemma` |
| C8 | DINOv2 ViT-S/14, blocks 3, 6, 9 | self-supervised ViT, small | as C5 | local statistics | this paper |
| C9 | CT dose network, blocks 1 to 3 | CT, task-trained (proposed) | dose classification on the train split | pointwise | this paper |
| C10 | CT dose network, blocks 1 to 3 | CT, task-trained (proposed) | as C9 | local statistics | this paper |

- **Screen only (no training):** DINOv2 ViT-L/14 (blocks 6, 12, 18 of 24, the
  same relative depths) and CLIP ViT-B/16, so the
  size and language axes have more points in Fig 2 and Fig 4 without the
  training cost.
- **Metrics (test split):** PSNR and SSIM with `data_range=1.0`; LPIPS with
  the AlexNet backbone (`lpips` package); DISTS (`piq`); high-band NPS ratio
  $R_H$. C2 and C3 are marked in Table 1 where they are scored with their own
  training metric (LPIPS-VGG trains, LPIPS-Alex scores, so C2 is only partly
  circular; C3 is fully circular on DISTS).

$$
\mathrm{PSNR}(\hat{x}, x) = 10 \log_{10} \frac{1}{\frac{1}{N}\sum_{i=1}^{N} (\hat{x}_i - x_i)^2}
$$

$$
R_H(\hat{x}, x) = \frac{\sum_{f \in \mathcal{B}_H} P_{\hat{x}}(f)}{\sum_{f \in \mathcal{B}_H} P_{x}(f)}, \qquad P_I(f) = \left| \mathcal{F}\{ I - \bar{I} \}(f) \right|^2
$$

where $\mathcal{F}$ is the 2D DFT, $\bar{I}$ the image mean and
$\mathcal{B}_H = \{ f : 0.25 \le \lVert f \rVert \le 0.5 \}$ cycles per pixel
(TODO: take the band edges from `viz_toolbox_roi_robustness`, as the
sibling plans do). $R_H = 1$ keeps the reference's fine texture; below 1
means smoothing, above 1 means added grain. Tests use $\lvert 1 - R_H \rvert$.

- **Seeds:** 42, 123, 456.
- **Statistics:** paired Wilcoxon signed-rank test on seed-averaged per-slice
  metrics. Family A: each of C1 to C10 vs C0 on PSNR, LPIPS and
  $\lvert 1 - R_H \rvert$, Holm over 30 comparisons. Family B (planned
  pairs): C5 vs C8 (matching), C9 vs C10 (matching), C5 vs C6 (size), on the
  same 3 metrics, Holm over 9. Matched-pairs rank-biserial effect size. Screen
  vs outcome: Spearman correlation across the 10 trained networks with a
  permutation p-value (n is small, so it is reported as exploratory).

### Compute Budget

| Item | Count |
|---|---|
| Trained configurations | 11 |
| Seeds | 3 |
| Full U-Net runs | 33 |
| CT dose network pretraining | 1 (about 1 GPU hour, TODO measure) |
| Screen (Step 1.4), forward passes only | 12 networks × 2 matchings × 182 val slices × 7 conditions |
| GPU hours per U-Net run, C0 (measured in Step 1.2) | - |
| GPU hours per U-Net run, C7 MedSigLIP (measured in Step 2.3) | - |
| Total GPU hours | - |
| GPU hours available in Weeks 1 to 3 (ozan + kiymet lab GPUs, Colab) | - |

**Cut order if over budget:** drop C6 seeds 123 and 456 (report seed 42 and
say so), then drop C4, then run C7 at batch 4 with 2-step accumulation on
Colab. Never drop seeds for C0, C5, C8, C9 or C10, because RQ3 rests on those
pairs.

## GitHub Issue Reporting

All experiment outputs must be documented as Markdown comments on the
corresponding GitHub issue for that week or milestone. A presentation may use
that issue comment directly or use a separate slide deck, but completed result
tables, figures or artifact links, commit hash, environment, data split, and
commands must still be posted in the issue comment.

---

## Week 1: Data, Reference and the Training-Free Screen (Mon Sep 28 to Sun Oct 4)

**Objective:** verified data and metrics, Arda's earlier runs audited, the L1
reference trained for seed 42, every feature network wrapped behind one
interface, and the screen (Fig 2) run before any perceptual training.

### Step 1.1: Shared Code: Config, Split Check, Loader and Metrics

**Produces:** `src/config.py`, `src/data.py`, `src/metrics.py`, used by every
later step. Nothing in the paper yet.

- [ ] Create the config and check keys, shapes and value range against the
      dataset README
- [ ] Run the patient leakage check and record patients per split
- [ ] Implement PSNR, SSIM, LPIPS-Alex, DISTS and $R_H$; pass the identity
      test
- [ ] Request access to `google/medsiglip-448` on Hugging Face today (gated
      model; Batuhan's project needs the same)

```python
# src/config.py
from dataclasses import dataclass
from pathlib import Path

DOSE_KEYS: tuple[str, ...] = ("noisy_clipped_10k", "noisy_clipped_50k", "noisy_clipped_100k",
                              "noisy_clipped_500k", "noisy_clipped_1000k")


@dataclass(frozen=True)
class PerceptualConfig:
    run_name: str
    data_root: Path
    feature_net: str = "none"
    matching: str = "pointwise"
    loss_weight: float = 0.0
    output_dir: Path = Path("outputs/perceptual_backbone")
    input_key: str = "noisy_clipped_100k"
    target_key: str = "gt_clipped"
    crop_size: int = 224
    seed: int = 42
    batch_size: int = 8
    grad_accumulation: int = 1
    lr: float = 1e-4
    epochs: int = 30
    use_amp: bool = True

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name
```

```python
# src/data.py
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
```

$$
\mathrm{SSIM}(a, b) = \frac{(2\mu_a\mu_b + c_1)(2\sigma_{ab} + c_2)}{(\mu_a^2 + \mu_b^2 + c_1)(\sigma_a^2 + \sigma_b^2 + c_2)}
$$

```python
# src/metrics.py
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import piq
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

LPIPS_ALEX = lpips.LPIPS(net="alex").eval()
DISTS_METRIC = piq.DISTS(reduction="none").eval()


def high_band_ratio(prediction: np.ndarray, target: np.ndarray, low: float = 0.25, high: float = 0.5) -> float:
    def band_power(image: np.ndarray) -> float:
        power = np.abs(np.fft.fft2(image - image.mean())) ** 2
        fy, fx = np.meshgrid(np.fft.fftfreq(image.shape[0]), np.fft.fftfreq(image.shape[1]), indexing="ij")
        radius = np.sqrt(fx ** 2 + fy ** 2)
        return float(power[(radius >= low) & (radius <= high)].sum())

    return band_power(prediction) / band_power(target)


@torch.no_grad()
def slice_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, ref = (torch.from_numpy(z).float()[None, None].expand(1, 3, *z.shape) for z in (prediction, target))
    return {
        "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=1.0)),
        "ssim": float(structural_similarity(target, prediction, data_range=1.0)),
        "lpips": float(LPIPS_ALEX(pred * 2 - 1, ref * 2 - 1)),
        "dists": float(DISTS_METRIC(pred.clamp(0, 1), ref.clamp(0, 1))),
        "r_h": high_band_ratio(prediction, target),
    }


def save_per_slice_metrics(rows: list[dict[str, float | str | int]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
```

Each CSV row holds `configuration`, `seed`, `slice_id`, `patient_id` and the
metric values.

| Check | Expected | Observed |
|---|---|---|
| Train / val / test patients | TODO | - |
| Shared patients between splits | 0 | - |
| Input and target range | [0, 1] | - |
| Metrics on identity input | PSNR inf, SSIM 1, LPIPS 0, DISTS 0, $R_H$ 1 | - |
| SSIM of `noisy_clipped_100k` vs `gt_clipped`, test mean | above the U-Net outputs, not below | - |

**Gotcha:** always pass `data_range=1.0` to SSIM. Arda's #261 runs report
output SSIM below the noisy input's SSIM (0.679 and 0.488 vs 0.750); that
pattern usually means a wrong data range or misaligned tensors, and Step 1.2
must explain it before any of those numbers are reused.

### Step 1.2: Audit Earlier Runs and Train the L1 Reference

**Produces:** Table 1 rows "LDCT input" and C0 (seed 42); GPU hours per run
in the Compute Budget; a decision on which of Arda's runs can be reused.

- [ ] For each earlier run, record dose level, split, epochs, crop size and
      whether a checkpoint exists
- [ ] Re-evaluate every existing checkpoint with `src/metrics.py`; a run is
      reusable only if its protocol matches the Protocol section
- [ ] Train C0 (L1) for seed 42, time it, fill the Compute Budget
- [ ] Explain the SSIM drop (data range, alignment, or real hallucination)

| Earlier run (source) | Reported PSNR (dB) | Reported SSIM | Dose known | Protocol matches | Action |
|---|---|---|---|---|---|
| L1 + 0.1·LPIPS (#208 comment 4) | 30.76 ± 2.09 | - | no | - | re-evaluate or rerun as C2 |
| L1 + 0.1·DISTS (#208 comment 4) | 30.58 ± 1.92 | - | no | - | re-evaluate or rerun as C3 |
| L1 + λ·DISTS on RadImageNet ResNet-50, λ 0.3 to 2 (#208 comment 7) | images only | - | no | - | recover the weights source for C4 |
| L1 + 2·DINOv2-S perceptual, 30 epochs (#261 comment 3) | 30.71 | 0.679 (input 0.750) | no | - | audit SSIM, then rerun as C5 |
| L1 + 2·DINOv2-S + 0.5·LPIPS (#261 comment 3) | 29.12 | 0.488 (input 0.746) | no | - | out of scope (mixed loss) |
| SwinIR, L1 + 3·DINOv2 + 0.3·LPIPS (#261 comment 3) | 31.46 | 0.880 | no | - | out of scope (different network) |
| Multi-scale vs single-scale LPIPS, λ 0.1 (#208 comment 4) | +7.46 vs +7.52 dB over input | - | no | - | out of scope |

The ± values in #208 are spreads over test slices, not seeds.

```python
# src/train.py
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
```

The perceptual term runs in float32 (`prediction.float()`) because ViT
features and DISTS statistics lose precision in float16. The full loop
pattern is in `wiki/guides/training_patterns.md`.

| Configuration (seed 42, test, 100k) | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | $R_H$ (→1) |
|---|---|---|---|---|---|
| LDCT input (100k) | - | - | - | - | - |
| C0 L1 | - | - | - | - | - |

**Gotcha:** the lab's earlier L1 U-Net at 100k reached PSNR 29.998 dB and
SSIM 0.884 (`viz_toolbox_roi_robustness`); C0 far from that means the
config drifted from the siblings.

### Step 1.3: Feature Network Zoo and the Two Distances

**Produces:** `src/feature_nets.py`, `src/distances.py`; equations (1) and (2)
in Section 2 of the paper.

- [ ] Wrap every network so it returns a list of dense feature maps
      $[B, C_l, h_l, w_l]$ from its chosen layers
- [ ] Implement the pointwise distance and the local statistics distance
- [ ] Check that both distances are 0 at equality and have gradients

Pointwise distance on channel-normalized features $\bar{\phi}_l$:

$$
d^{\mathrm{pt}}_\phi(\hat{x}, x) = \sum_{l \in \mathcal{L}_\phi} \frac{1}{H_l W_l} \sum_{h, w} \big\lVert \bar{\phi}_l(\hat{x})_{hw} - \bar{\phi}_l(x)_{hw} \big\rVert_2^2 \tag{1}
$$

Local statistics distance: for each channel $c$ and each $k \times k$ window
$\omega$ of the feature map, the mean $\mu$ and standard deviation $\sigma$
of the prediction (hat) and target are compared, with no cross-correlation
term:

$$
d^{\mathrm{st}}_\phi(\hat{x}, x) = \sum_{l \in \mathcal{L}_\phi} \frac{1}{C_l \lvert \Omega_l \rvert} \sum_{c, \omega} \left[ 1 - \frac{2 \hat{\mu} \mu + c_1}{\hat{\mu}^2 + \mu^2 + c_1} \cdot \frac{2 \hat{\sigma} \sigma + c_2}{\hat{\sigma}^2 + \sigma^2 + c_2} \right] \tag{2}
$$

Normal-dose noise is random, so the output cannot match the reference grain
pixel by pixel; (2) asks only that each local region has the same feature
mean and spread, which is the texture idea behind DISTS
(`ding2020dists`) applied to any network. $c_1 = c_2 = 10^{-6}$, $k = 4$.

```python
# src/feature_nets.py
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
```

```python
# src/distances.py
import piq
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.feature_nets import FeatureNet


def normalize_channels(feature: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return feature / (feature.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)


def pointwise_distance(predicted: list[torch.Tensor], reference: list[torch.Tensor]) -> torch.Tensor:
    return sum((normalize_channels(p) - normalize_channels(r)).pow(2).sum(dim=1).mean() for p, r in zip(predicted, reference))


def local_stats(feature: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = F.avg_pool2d(feature, window)
    variance = (F.avg_pool2d(feature.pow(2), window) - mean.pow(2)).clamp(min=0)
    return mean, variance.sqrt()


def statistics_distance(predicted: list[torch.Tensor], reference: list[torch.Tensor], window: int = 4,
                        c1: float = 1e-6, c2: float = 1e-6) -> torch.Tensor:
    total = predicted[0].new_zeros(())
    for p, r in zip(predicted, reference):
        (mp, sp), (mr, sr) = local_stats(normalize_channels(p), window), local_stats(normalize_channels(r), window)
        similarity = (2 * mp * mr + c1) / (mp.pow(2) + mr.pow(2) + c1) * (2 * sp * sr + c2) / (sp.pow(2) + sr.pow(2) + c2)
        total = total + (1 - similarity).mean()
    return total


class FeatureDistance(nn.Module):
    def __init__(self, net: FeatureNet, matching: str) -> None:
        super().__init__()
        self.net = net.eval().requires_grad_(False)
        self.distance = pointwise_distance if matching == "pointwise" else statistics_distance

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reference = self.net.features(target)
        return self.distance(self.net.features(prediction), reference)


class PiqDistance(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.metric = (piq.LPIPS(reduction="mean") if name == "lpips_vgg" else piq.DISTS(reduction="mean")).eval()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.metric(prediction.clamp(0, 1).expand(-1, 3, -1, -1), target.expand(-1, 3, -1, -1))


def shared_crop(prediction: torch.Tensor, target: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    top = torch.randint(0, prediction.shape[-2] - size + 1, (1,)).item()
    left = torch.randint(0, prediction.shape[-1] - size + 1, (1,)).item()
    return prediction[..., top:top + size, left:left + size], target[..., top:top + size, left:left + size]


class PerceptualObjective(nn.Module):
    def __init__(self, distance: nn.Module | None, weight: float, crop_size: int = 224) -> None:
        super().__init__()
        self.distance = distance
        self.weight = weight
        self.crop_size = crop_size

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(prediction, target)
        if self.distance is None or self.weight == 0:
            return loss
        return loss + self.weight * self.distance(*shared_crop(prediction, target, self.crop_size))
```

| Check (random 64 × 64 input, every network, both matchings) | Expected | Observed |
|---|---|---|
| Distance at prediction = target | 0 | - |
| Gradient reaches the prediction | yes | - |
| ms per batch of 8 at 224 × 224 (forward and backward) | recorded per network | - |

**Gotcha:** `strict=False` hides missing RadImageNet keys; print the missing
and unexpected keys once and make sure only the classifier head is missing.
MedSigLIP has no CLS token (`has_cls=False`); CLIP does.

### Step 1.4: Training-Free Screen: Dose Ordering and the Equal-MSE Blur Test

**Produces:** Fig 2 (a) and (b); the screen columns of Table 2.

- [ ] For every network in the zoo (C1 to C8 networks, DINOv2-L, CLIP) and
      both matchings, compute the distance between each val slice's normal
      dose image and its 5 low-dose versions
- [ ] For every val slice, find the Gaussian blur $\sigma_i$ whose MSE to the
      normal-dose image equals the MSE of the 100k image, and compute the blur
      penalty
- [ ] Add the pixel L1 as a reference line (its blur penalty is close to 1 by
      construction)
- [ ] Repeat for the CT dose network after Step 2.2

Dose curve and ordering score, with $x^{(d)}$ the slice at dose $d$:

$$
D_\phi(d) = \frac{1}{N} \sum_{i=1}^{N} d_\phi\big(x_i^{(d)}, x_i\big), \qquad M_\phi = \frac{1}{N} \sum_{i=1}^{N} \rho_{\mathrm{S}}\Big( \big[d_\phi(x_i^{(d)}, x_i)\big]_{d}, \; \big[\log(1/d)\big]_{d} \Big)
$$

where $\rho_{\mathrm{S}}$ is Spearman's correlation over the 5 doses
($M_\phi = 1$: the distance always grows as dose falls).

Equal-MSE blur penalty, with $G_{\sigma}$ a Gaussian blur and $\sigma_i$
chosen by bisection so that
$\lVert G_{\sigma_i} x_i - x_i \rVert_2^2 = \lVert x_i^{(100k)} - x_i \rVert_2^2$:

$$
B_\phi = \frac{\sum_i d_\phi\big(G_{\sigma_i} x_i, x_i\big)}{\sum_i d_\phi\big(x_i^{(100k)}, x_i\big)}
$$

$B_\phi \ge 1$ means the network penalizes blur at least as much as noise of
the same pixel error; $B_\phi < 1$ means a denoiser trained with it can lower
the loss by smoothing (the failure the lab's MedSigLIP IQA model shows).

```python
# scripts/screen.py
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr
from tqdm import tqdm

from src.config import DOSE_KEYS
from src.data import load_slice
from src.distances import FeatureDistance

DOSES: tuple[int, ...] = (10_000, 50_000, 100_000, 500_000, 1_000_000)


def matched_blur(clean: torch.Tensor, noisy: torch.Tensor, steps: int = 20) -> torch.Tensor:
    target_mse, low, high = torch.mean((noisy - clean) ** 2).item(), 0.05, 8.0
    for _ in range(steps):
        sigma = (low + high) / 2
        blurred = TF.gaussian_blur(clean, kernel_size=2 * int(4 * sigma) + 1, sigma=sigma)
        low, high = (sigma, high) if torch.mean((blurred - clean) ** 2).item() < target_mse else (low, sigma)
    return blurred


@torch.no_grad()
def screen_network(distance: FeatureDistance, files: list[Path], device: torch.device,
                   crop: int = 224) -> pd.DataFrame:
    rows = []
    for path in tqdm(files, desc="screen", leave=False):
        clean = load_slice(path, "gt_clipped")[None, :, 144:144 + crop, 144:144 + crop].to(device)
        noisy = {dose: load_slice(path, key)[None, :, 144:144 + crop, 144:144 + crop].to(device)
                 for dose, key in zip(DOSES, DOSE_KEYS)}
        distances = [distance(noisy[dose], clean).item() for dose in DOSES]
        rows.append({
            "slice_id": path.stem,
            "ordering": spearmanr(distances, np.log(1 / np.array(DOSES))).statistic,
            **{f"d_{dose}": value for dose, value in zip(DOSES, distances)},
            "d_blur": distance(matched_blur(clean, noisy[100_000]), clean).item(),
        })
    frame = pd.DataFrame(rows)
    frame.attrs["blur_penalty"] = frame["d_blur"].sum() / frame["d_100000"].sum()
    return frame
```

The fixed centre crop (rows and columns 144 to 368) keeps the screen
deterministic; it lies inside the body on I2I slices (check a few by eye).

| Network | Matching | Ordering $M_\phi$ ↑ | $D_\phi(10k) / D_\phi(1000k)$ ↑ | Blur penalty $B_\phi$ (≥ 1) | Parameters |
|---|---|---|---|---|---|
| Pixel L1 (reference) | - | - | - | - | 0 |
| VGG-19 relu3_2 | pointwise | - | - | - | - |
| LPIPS-VGG | pointwise | - | - | - | - |
| DISTS | statistics | - | - | - | - |
| RadImageNet ResNet-50 | pointwise | - | - | - | - |
| DINOv2-S | pointwise | - | - | - | - |
| DINOv2-S | statistics | - | - | - | - |
| DINOv2-B | pointwise | - | - | - | - |
| DINOv2-L | pointwise | - | - | - | - |
| CLIP ViT-B/16 | pointwise | - | - | - | - |
| MedSigLIP-448 | pointwise | - | - | - | - |
| CT dose network (Step 2.2) | pointwise | - | - | - | - |
| CT dose network (Step 2.2) | statistics | - | - | - | - |

**Gotcha:** the screen is a prediction, not a filter. Every C1 to C10 network
is trained regardless of its score; otherwise Fig 4 cannot test whether the
screen predicts the outcome.

### Week 1 Summary

- [ ] **Checkpoint (Sun Oct 4):** C0 row filled and close to the sibling L1,
      earlier runs audited, SSIM issue explained, screen table filled for the
      pretrained networks, Compute Budget filled
- [ ] Post the week's tables, Fig 2 draft, key result, takeaway, commit hash,
      environment, data split, and commands as a Markdown comment on the
      GitHub issue.

**Key Result:** <to be filled>
**Takeaway:** <to be filled>

---

## Week 2: Perceptual Losses and the CT Dose Network (Mon Oct 5 to Sun Oct 11)

**Objective:** loss weights calibrated, the CT dose network trained and
screened, and all 10 perceptual configurations trained for seed 42.

### Step 2.1: Calibrate Each Loss Weight by Perceptual Influence

**Produces:** the $\lambda_\phi$ column of Table 2.

- [ ] For each configuration, run the untrained U-Net (seed 42) over the
      training set once and sum the L1 term and the feature distance
- [ ] Set $\lambda_\phi$ so the perceptual term is 95% of the initial loss,
      as in `viana2025perceptual`
- [ ] State in the paper that the weights are matched by influence, not tuned
      per network

$$
\Psi(\lambda) = \frac{\lambda S_\phi}{S_1 + \lambda S_\phi} \quad \Rightarrow \quad \lambda_\phi = \frac{\Psi^\star}{1 - \Psi^\star} \cdot \frac{S_1}{S_\phi}, \qquad \Psi^\star = 0.95
$$

where $S_1 = \sum_i \lVert f_{\theta_0}(x_{\mathrm{LD}, i}) - x_i \rVert_1$
and $S_\phi = \sum_i d_\phi(\cdot)$ over the training set at initialization
$\theta_0$ (Viana et al. use MSE in place of L1; we keep L1 as the pixel term
for every configuration).

```python
# scripts/calibrate_lambda.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.distances import shared_crop


@torch.no_grad()
def calibrate_weight(model: nn.Module, distance: nn.Module, loader: DataLoader, device: torch.device,
                     target_influence: float = 0.95, crop_size: int = 224) -> float:
    torch.manual_seed(42)
    model.eval()
    pixel_sum, feature_sum = 0.0, 0.0
    for noisy, clean, _ in tqdm(loader, desc="calibrate", leave=False):
        prediction, clean = model(noisy.to(device)), clean.to(device)
        pixel_sum += F.l1_loss(prediction, clean).item()
        feature_sum += distance(*shared_crop(prediction, clean, crop_size)).item()
    return target_influence / (1 - target_influence) * pixel_sum / feature_sum
```

| Configuration | $S_1$ | $S_\phi$ | $\lambda_\phi$ |
|---|---|---|---|
| C1 VGG-19 | - | - | - |
| C2 LPIPS-VGG | - | - | - |
| C3 DISTS | - | - | - |
| C4 RadImageNet | - | - | - |
| C5 DINOv2-S pointwise | - | - | - |
| C6 DINOv2-B pointwise | - | - | - |
| C7 MedSigLIP | - | - | - |
| C8 DINOv2-S statistics | - | - | - |
| C9 CT dose net pointwise | - | - | - |
| C10 CT dose net statistics | - | - | - |

**Gotcha:** calibrate with the same seed-42 initialization and crops for every
network, otherwise $S_1$ differs between rows. If one $\Psi^\star = 0.95$
configuration diverges in its first epoch, lower $\Psi^\star$ for **all**
configurations and recalibrate, never for one row only.

### Step 2.2: Train the CT Dose Network

**Produces:** `src/ct_dose_net.py`, the CT dose network weights, its rows in
the Step 1.4 screen table, and the method paragraph 2.3 of the paper.

- [ ] Build 128 × 128 patches from **train-split patients only**, labelled by
      source: the 5 dose keys and `gt_clipped` (6 classes)
- [ ] Train 10 epochs with cross-entropy; report val accuracy and the
      confusion matrix
- [ ] Freeze it and run the Step 1.4 screen on it (both matchings)

$$
\mathcal{L}_{\mathrm{dose}} = -\frac{1}{B} \sum_{b=1}^{B} \log \operatorname{softmax}\big(g_\psi(p_b)\big)_{y_b}, \qquad y_b \in \{10\mathrm{k}, 50\mathrm{k}, 100\mathrm{k}, 500\mathrm{k}, 1000\mathrm{k}, \mathrm{clean}\}
$$

where $p_b$ is a patch and $g_\psi$ the network; its block outputs are the
features $\phi_l$ used in (1) and (2).

```python
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
```

`DoseNetFeatures` takes the 1-channel crop directly (it does not call
`prepare`). Sample patches with random positions inside the body mask and
train with Adam, learning rate $10^{-3}$, batch 64.

| Check | Expected | Observed |
|---|---|---|
| Val accuracy, 6 classes | well above 1/6; 10k and clean easy, 500k vs 1000k hardest | - |
| Val accuracy, clean vs 1000k | above chance | - |
| Screen ordering $M_\phi$ | close to 1 | - |
| Screen blur penalty $B_\phi$ | report; below 1 is a finding, not a bug | - |

**Gotcha:** a network trained only to tell noise levels apart may treat a
blurred slice as "clean" (the same smoothness bias as the MedSigLIP IQA
model). Do not add blurred patches to its training to fix the screen, because
that tunes it to the test. Report $B_\phi$ as measured and let C10 (local
statistics) show whether matching the texture spread avoids the problem.

### Step 2.3: Seed 42 Runs for C1 to C10 and the Fig 1 Draft

**Produces:** Table 1 rows C1 to C10 (seed 42); per-network time per step in
the Compute Budget; Fig 1 draft.

- [ ] Train C1 to C10 with the Step 2.1 weights and the Protocol budget
- [ ] Run C7 (MedSigLIP) on the GPU with the most memory; use
      `batch_size=4, grad_accumulation=2` if needed
- [ ] Draft Fig 1 in `figures/fig1_overview.drawio`: feature zoo grouped by
      family on the left, the training-free screen in the middle, the
      $\Psi$-matched U-Net training on the right, evaluation metrics below

```python
# scripts/run_config.py
from pathlib import Path

import pandas as pd
import torch

from src.config import PerceptualConfig
from src.ct_dose_net import CTDoseNet, DoseNetFeatures
from src.distances import FeatureDistance, PerceptualObjective, PiqDistance
from src.feature_nets import build_feature_net
from src.train import train_and_test
from src.unet import UNet  # TODO: the lab U-Net shared with tissue_band_nps_loss


def build_objective(config: PerceptualConfig, dose_net_path: Path, radimagenet: Path | None) -> PerceptualObjective:
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
    return PerceptualObjective(FeatureDistance(net, config.matching), config.loss_weight, config.crop_size)


def run(config: PerceptualConfig, dose_net_path: Path, radimagenet: Path | None = None) -> pd.DataFrame:
    device = torch.device("cuda")
    objective = build_objective(config, dose_net_path, radimagenet).to(device)
    return train_and_test(config, UNet().to(device), objective, device)
```

| Configuration (seed 42, test, 100k) | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | $R_H$ (→1) | ms per step |
|---|---|---|---|---|---|---|
| C0 L1 (from Step 1.2) | - | - | - | - | - | - |
| C1 VGG-19 | - | - | - | - | - | - |
| C2 LPIPS-VGG | - | - | - | - | - | - |
| C3 DISTS | - | - | - | - | - | - |
| C4 RadImageNet | - | - | - | - | - | - |
| C5 DINOv2-S pointwise | - | - | - | - | - | - |
| C6 DINOv2-B pointwise | - | - | - | - | - | - |
| C7 MedSigLIP | - | - | - | - | - | - |
| C8 DINOv2-S statistics | - | - | - | - | - | - |
| C9 CT dose net pointwise | - | - | - | - | - | - |
| C10 CT dose net statistics | - | - | - | - | - | - |

**Gotcha:** pure perceptual losses hallucinated texture in Arda's #208 runs;
here the L1 term is always present and $\Psi$ is matched. If a configuration
still shows structures absent from the target, keep it in the table and show
it in Fig 3; that is a result.

### Week 2 Summary

- [ ] **Checkpoint (Sun Oct 11):** every configuration has a seed 42 row,
      Table 2 weights filled, CT dose network screened, Fig 1 drafted; go or
      no-go for Week 3 with the advisor
- [ ] Post the week's results as a Markdown comment on the GitHub issue.

**Key Result:** <to be filled>
**Takeaway:** <to be filled>

---

## Week 3: Seeds, Statistics and Figures (Mon Oct 12 to Sun Oct 18)

**Objective:** 3-seed results for all 11 configurations, both test families,
the screen-vs-outcome test, and every figure generated from scripts, frozen
on Sun Oct 18.

### Step 3.1: Seeds 123 and 456

**Produces:** the ± std in Table 1; `outputs/perceptual_backbone/per_slice_metrics.csv`.

- [ ] Run seeds 123 and 456 for C0 to C10 (the Step 2.1 weights stay fixed)
- [ ] Concatenate every run's metrics into one per-slice CSV

```python
# scripts/run_all_seeds.py
import random
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from src.config import PerceptualConfig

SEEDS: tuple[int, ...] = (42, 123, 456)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_all_seeds(configurations: dict[str, PerceptualConfig],
                  train_and_test: Callable[[PerceptualConfig], pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, base in configurations.items():
        for seed in SEEDS:
            config = replace(base, run_name=f"{name}_seed{seed}", seed=seed)
            cached = config.run_dir / "metrics.csv"
            if cached.exists():
                frame = pd.read_csv(cached)
            else:
                set_seed(seed)
                frame = train_and_test(config)
            frames.append(frame.assign(configuration=name, seed=seed))
    return pd.concat(frames, ignore_index=True)
```

### Step 3.2: Statistics and the Screen-vs-Outcome Test

**Produces:** † marks in Table 1, the Family B table, Fig 4, the abstract
number.

- [ ] Family A: each of C1 to C10 vs C0 on PSNR, LPIPS and
      $\lvert 1 - R_H \rvert$, Holm over 30
- [ ] Family B: C5 vs C8, C9 vs C10, C5 vs C6 on the same metrics, Holm over 9
- [ ] Spearman correlation across the 10 trained networks between screen
      scores ($M_\phi$, $B_\phi$) and trained outcomes (DISTS, LPIPS,
      $\lvert 1 - R_H \rvert$), with a permutation p-value
- [ ] Fix both families now; do not change them after seeing p-values

$$
W = \sum_{i=1}^{n} \mathrm{sgn}(d_i) R_i, \qquad r = \frac{\sum_{d_i > 0} R_i - \sum_{d_i < 0} R_i}{\sum_i R_i}
$$

with $d_i$ the per-slice difference of the seed-averaged metric and $R_i$ the
rank of $\lvert d_i \rvert$.

```python
# scripts/statistics.py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted.tolist()


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    ranks = rankdata(np.abs(nonzero))
    positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative)) if len(nonzero) else 0.0


def paired_family(per_slice: pd.DataFrame, pairs: list[tuple[str, str]], metrics: list[str]) -> pd.DataFrame:
    averaged = per_slice.groupby(["configuration", "slice_id"])[metrics].mean().reset_index()
    rows = []
    for metric in metrics:
        wide = averaged.pivot(index="slice_id", columns="configuration", values=metric)
        for first, second in pairs:
            differences = (wide[first] - wide[second]).dropna().to_numpy()
            rows.append({"comparison": f"{first} vs {second}", "metric": metric,
                         "median_difference": float(np.median(differences)),
                         "p_raw": float(wilcoxon(differences).pvalue), "effect_size_r": rank_biserial(differences)})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"].tolist())
    return result


def screen_predicts_outcome(screen: pd.Series, outcome: pd.Series, permutations: int = 10_000,
                            seed: int = 42) -> tuple[float, float]:
    rho = spearmanr(screen, outcome).statistic
    rng = np.random.default_rng(seed)
    null = [spearmanr(screen, rng.permutation(outcome.to_numpy())).statistic for _ in range(permutations)]
    return float(rho), float(np.mean(np.abs(null) >= abs(rho)))
```

Add `r_h_error = abs(1 - r_h)` before calling. Family A pairs are
`[(f"C{k}", "C0") for k in range(1, 11)]`; Family B pairs are
`[("C8", "C5"), ("C10", "C9"), ("C6", "C5")]`.

| Family B comparison | Metric | Median difference | p (Holm over 9) | Effect size r |
|---|---|---|---|---|
| C8 vs C5 (statistics vs pointwise, DINOv2-S) | DISTS | - | - | - |
| C8 vs C5 | $\lvert 1 - R_H \rvert$ | - | - | - |
| C10 vs C9 (statistics vs pointwise, CT dose net) | DISTS | - | - | - |
| C10 vs C9 | $\lvert 1 - R_H \rvert$ | - | - | - |
| C6 vs C5 (base vs small) | DISTS | - | - | - |
| C6 vs C5 | $\lvert 1 - R_H \rvert$ | - | - | - |
| <remaining 3 rows (PSNR) stay in the CSV> | - | - | - | - |

| Screen score vs trained outcome (10 networks) | Spearman $\rho$ | Permutation p |
|---|---|---|
| $B_\phi$ vs $\lvert 1 - R_H \rvert$ | - | - |
| $B_\phi$ vs DISTS | - | - |
| $M_\phi$ vs $\lvert 1 - R_H \rvert$ | - | - |
| Parameters vs DISTS | - | - |

**Gotcha:** DISTS is circular for C3 and partly for C2; compute the screen
correlation both with and without C2 and C3 and report both.

### Step 3.3: Figures and Table Export

**Produces:** Fig 2, Fig 3, Fig 4, final Fig 1; Markdown and LaTeX for Table 1
and Table 2.

- [ ] Fig 2 (`scripts/make_fig2_screen.py`): (a) $D_\phi(d)$ normalized by
      $D_\phi(10k)$ against dose on a log axis, one line per network, pixel
      L1 dashed; (b) $B_\phi$ bars per network, pointwise and statistics side
      by side, a line at 1
- [ ] Fig 3 (`scripts/make_fig3_qualitative.py`): LDCT, C0, C1, C3, C5, C7,
      C10, NDCT; the 2 test slices closest to the median C0 PSNR (not chosen
      by eye); one soft-tissue zoom; one error scale
- [ ] Fig 4 (`scripts/make_fig4_screen_vs_outcome.py`): $B_\phi$ against
      trained $\lvert 1 - R_H \rvert$, marker size by parameter count,
      colour by family, DINOv2 S and B joined by a line, $\rho$ in the corner
- [ ] Finalize Fig 1; export every figure as PDF
- [ ] Run `scripts/export_tables.py` and paste its Markdown output into Final
      Tables

```python
# scripts/make_fig3_qualitative.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def qualitative_grid(images: dict[str, np.ndarray], reference: str, roi: tuple[int, int, int], path: Path,
                     error_max: float = 0.1) -> None:
    row, col, size = roi
    names = list(images)
    figure, axes = plt.subplots(2, len(names), figsize=(7.0, 2.4), constrained_layout=True)
    for index, name in enumerate(names):
        top, bottom = axes[0, index], axes[1, index]
        top.imshow(images[name], cmap="gray", vmin=0, vmax=1)
        top.add_patch(Rectangle((col, row), size, size, fill=False, edgecolor="yellow", linewidth=0.8))
        inset = top.inset_axes([0.58, 0.58, 0.4, 0.4])
        inset.imshow(images[name][row:row + size, col:col + size], cmap="gray", vmin=0, vmax=1)
        inset.set_xticks([])
        inset.set_yticks([])
        top.set_title(name, fontsize=6)
        bottom.imshow(np.abs(images[name] - images[reference]), cmap="magma", vmin=0, vmax=error_max)
        for axis in (top, bottom):
            axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
```

```python
# scripts/export_tables.py
import pandas as pd


def formatted_cells(per_run: pd.DataFrame, metrics: dict[str, bool], order: list[str],
                    decimals: int = 3) -> dict[str, list[tuple[str, bool]]]:
    stats = per_run.groupby("configuration")[list(metrics)].agg(["mean", "std"])
    best = {m: (stats[(m, "mean")].idxmax() if higher else stats[(m, "mean")].idxmin()) for m, higher in metrics.items()}
    return {
        name: [(f"{stats.loc[name, (m, 'mean')]:.{decimals}f} ± {stats.loc[name, (m, 'std')]:.{decimals}f}",
                best[m] == name) for m in metrics]
        for name in order
    }


def markdown_results_table(per_run: pd.DataFrame, metrics: dict[str, bool], order: list[str]) -> str:
    arrows = {True: "↑", False: "↓"}
    lines = ["| Configuration | " + " | ".join(f"{m} {arrows[h]}" for m, h in metrics.items()) + " |",
             "|---|" + "---|" * len(metrics)]
    for name, cells in formatted_cells(per_run, metrics, order).items():
        lines.append(f"| {name} | " + " | ".join(f"**{v}**" if is_best else v for v, is_best in cells) + " |")
    return "\n".join(lines)


def latex_results_table(per_run: pd.DataFrame, metrics: dict[str, bool], order: list[str]) -> str:
    arrows = {True: "$\\uparrow$", False: "$\\downarrow$"}
    header = " & ".join(["Configuration"] + [f"{m} {arrows[h]}" for m, h in metrics.items()])
    lines = ["\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule", header + " \\\\", "\\midrule"]
    for name, cells in formatted_cells(per_run, metrics, order).items():
        values = [v.replace("±", "$\\pm$") for v, _ in cells]
        values = [f"\\textbf{{{v}}}" if is_best else v for v, (_, is_best) in zip(values, cells)]
        lines.append(" & ".join([name] + values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)
```

`per_run` is the per-slice CSV averaged to one row per configuration and seed,
so the ± is the spread across seeds. Pass `r_h_error` with `higher=False`
for the bold rule, and show $R_H$ itself in the table. Add † to a
configuration name when its Family A Holm-adjusted p < 0.05 on that metric,
and * where the metric is the configuration's own training loss.

### Week 3 Summary

- [ ] **Checkpoint (Sun Oct 18): results freeze.** Final Tables filled,
      figures exported as PDF. Anything missing now is left out of the paper.
- [ ] Post the Final Tables, figures and commit hash as a Markdown comment on
      the GitHub issue.

**Key Result:** <to be filled; this sentence becomes the abstract number>
**Takeaway:** <to be filled>

---

## Final Tables

Generated by `scripts/export_tables.py` at the results freeze. Do not edit
numbers by hand.

**Table 1: <caption with the takeaway, for example which family wins texture
at no PSNR cost>** (test split, 100k, mean ± std over seeds 42, 123, 456;
† Holm-adjusted p < 0.05 vs C0 in Family A; * scored with its own training
metric)

| Configuration | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | $R_H$ (→1) |
|---|---|---|---|---|---|
| LDCT input (100k) | - | - | - | - | - |
| C0 L1 | - | - | - | - | - |
| C1 VGG-19 | - | - | - | - | - |
| C2 LPIPS-VGG | - | - | - | - | - |
| C3 DISTS | - | - | - | - | - |
| C4 RadImageNet | - | - | - | - | - |
| C5 DINOv2-S pointwise | - | - | - | - | - |
| C6 DINOv2-B pointwise | - | - | - | - | - |
| C7 MedSigLIP | - | - | - | - | - |
| C8 DINOv2-S statistics | - | - | - | - | - |
| C9 CT dose net pointwise | - | - | - | - | - |
| C10 CT dose net statistics | - | - | - | - | - |

**Table 2: Feature networks.** Family, parameters, pretraining data, layers,
input size, $\lambda_\phi$ (Step 2.1), ms per step (Step 2.3), and screen
scores $M_\phi$ and $B_\phi$ (Step 1.4). Filled from those steps; one row per
network in Table 1.

| Network | Family | Parameters | Pretraining | Layers | Input | $\lambda_\phi$ | ms per step | $M_\phi$ | $B_\phi$ |
|---|---|---|---|---|---|---|---|---|---|
| VGG-19 | ImageNet CNN | - | ImageNet | relu3_2 | 224 | - | - | - | - |
| LPIPS-VGG | calibrated CNN | - | ImageNet + BAPPS | 5 blocks | 224 | - | - | - | - |
| DISTS | texture statistics | - | ImageNet + KADID | 5 blocks | 224 | - | - | - | - |
| RadImageNet ResNet-50 | medical CNN | - | RadImageNet | layer1 to 3 | 224 | - | - | - | - |
| DINOv2-S | self-supervised ViT | - | LVD-142M | 3, 6, 9 | 224 | - | - | - | - |
| DINOv2-B | self-supervised ViT | - | LVD-142M | 3, 6, 9 | 224 | - | - | - | - |
| MedSigLIP-448 | medical VLM | - | medical image-text | 7, 14, 20 | 448 | - | - | - | - |
| CT dose network | CT task-trained | - | I2I train split | blocks 1 to 3 | 224 | - | - | - | - |

---

## Week 4: Writing (Mon Oct 19 to Sun Oct 25)

**Objective:** a complete 4-page first draft in 4 writing days, shared with
the team on Fri Oct 23, 3 days before the official deadline. No new
experiments this week.

| Day | Writing task |
|---|---|
| Mon Oct 19 | Figures and tables into the LaTeX template; Method and Experimental Setup |
| Tue Oct 20 | Results and Introduction |
| Wed Oct 21 | Abstract, Conclusion, Ethics and Acknowledgments |
| Thu Oct 22 | Page fit to 4 pages, reference audit, final checklist |
| Fri Oct 23 | **Share the first draft with the team** |

- [ ] Copy the lab ISBI template from `projects/_done/proj2proj/isbi/` (`root.tex`
      with one `.tex` file per section, `spconf.sty`, `IEEEbib.bst`,
      `tables/`, `images/`) into `projects/perceptual_backbone_ldct/paper/`;
      clear the old content and point the bibliography to `../index.bib`
- [ ] Create `index.bib` from the Missing list in
      [novelty.md](novelty.md#reference-status) if not done yet
- [ ] Mon Oct 19: re-run the search strings from novelty.md; add anything new
      to its Near Matches and to Introduction P4
- [ ] Mon Oct 19: Figures 1 to 4 and Tables 1 to 2 placed, captions state the
      takeaway; Method and Experimental Setup
- [ ] Tue Oct 20: Results and Introduction
- [ ] Wed Oct 21: Abstract, Conclusion, Ethics and Acknowledgments
- [ ] Thu Oct 22: page fit, Reference Audit and Figures and Tables Audit below
- [ ] **Checkpoint (Fri Oct 23): first draft shared with the team.** Post the
      PDF link and open questions as a Markdown comment on the GitHub issue.
- [ ] Sun Oct 25: team comments collected (weekend review, so the draft can
      be submitted on the official deadline if needed)

### Word Budget

| Section | Target words | Paragraphs | Notes |
|---|---|---|---|
| Title | 8 to 14 | - | research-question form |
| Abstract | 150 to 200 | 1 (6 to 8 sentences) | one number: best family's texture gain over L1 at its PSNR cost, or the screen correlation |
| Index terms | 4 to 5 terms | - | |
| 1 Introduction | 400 to 500 | 5 | Fig 1 on page 1, 3 contributions |
| 2 Method | 600 to 700 | 4 subsections | equations (1) and (2), $\Psi$, screen, dose network |
| 3 Experiments and Results | 500 to 650 | setup + 3 result paragraphs | Table 1, Fig 2, Fig 3, Fig 4 |
| 4 Conclusion | 100 to 150 | 3 short | one limitation |
| Ethics and Acknowledgments | about 40 | 2 | public data template |
| References | 15 to 20 entries | - | |
| **Total prose** | **about 2,000** | | plus about 170 words of captions |

## Final Checklist (Paragraph by Paragraph)

One checkbox per paragraph of the paper. **Tell** is the paragraph's one
message, **Cite** the bibkeys it must contain (details in
[novelty.md](novelty.md#related-work)), **Compare** the numbers it must
contrast (from [Final Tables](#final-tables)).

### Abstract (150 to 200 words)

- [ ] **S1 Context.** Tell: perceptual losses reduce the over-smoothing of
      LDCT denoisers, and the feature network they use is usually an ImageNet
      VGG chosen by habit.
- [ ] **S2 Gap.** Tell: "However, whether larger, self-supervised or medical
      vision-language networks make better perceptual losses for CT has not
      been tested under one protocol."
- [ ] **S3 Pivot.** Tell: "We compare nine feature networks from five
      families, and a small CT dose network, as perceptual losses for one
      U-Net with matched loss influence."
- [ ] **S4 Method.** Tell: a training-free screen (dose ordering, equal-MSE
      blur test) and pointwise vs local statistics matching.
- [ ] **S5 to S6 Result.** Compare: the best configuration vs L1 on
      $\lvert 1 - R_H \rvert$ and DISTS with PSNR cost and Holm p; the screen
      correlation $\rho$.
- [ ] **S7 Implication.** Tell: which feature network and matching to use for
      LDCT, and whether size helps.

### Index Terms

- [ ] Low-dose CT, denoising, perceptual loss, foundation models, noise
      texture

### 1 Introduction (400 to 500 words)

- [ ] **P1 Motivation (about 80 words).** Tell: pixel losses over-smooth LDCT
      and erase the texture radiologists read; perceptual losses are the
      common fix. Cite: `eulig2024benchmarking`, `yang2018wganvgg`.
- [ ] **P2 Perceptual losses for LDCT (about 100 words).** Tell: VGG is the
      default; CT-specific (SACNN), language-aligned (LEDA) and
      foundation-model (ALDEN, D-PerceptCT) features appear inside individual
      methods, never compared under one protocol. Cite: `li2020sacnn`,
      `chen2024leda`, `wang2025alden`, `taifour2025dperceptct`.
- [ ] **P3 Why bigger is not obviously better (about 100 words).** Tell:
      better classifiers are not better similarity metrics; a large CT quality
      model prefers blurred slices; domain features did not help retinal
      synthesis. Cite: `kumar2022imagenet`, `demiroglu2025medsiglip`,
      `skorniewska2025retinal`, `zhang2018lpips`, `ding2020dists`.
- [ ] **P4 Gap and proposal (about 90 words).** Tell: Viana et al. vary only
      VGG layer and pretraining set and report PSNR and SSIM; we compare
      network families with matched influence, texture and cross-scored
      metrics, and test a screen. Cite: `viana2025perceptual`. Compare: their
      4 VGG settings vs our 11 configurations. No sentence from "Claims we
      must not make" in novelty.md.
- [ ] **P5 Contributions (about 80 words).** Tell: the three contributions
      from the README, pointing to Table 1, Fig 2 and Fig 4.
- [ ] Fig 1 on page 1, referenced in P4 or P5.

### 2 Method (600 to 700 words)

- [ ] **2.1 Objective and feature distances.** Tell: the L1 + $\lambda_\phi
      d_\phi$ objective, equation (1) pointwise and equation (2) local
      statistics, and why random CT noise favours (2). Cite:
      `ding2020dists`, `zhang2018lpips`.
- [ ] **2.2 Feature networks.** Tell: the five families and the chosen layers
      (Table 2), and the shared 224 crop. Cite: `oquab2024dinov2`,
      `sellergren2025medgemma`, `mei2022radimagenet`, `radford2021clip`.
- [ ] **2.3 CT dose network.** Tell: a small CNN trained to classify 5 doses
      and clean on train-split patches; why its features must track noise.
      Compare: SACNN's autoencoder features. Cite: `li2020sacnn`.
- [ ] **2.4 Matched influence and the screen.** Tell: $\lambda_\phi$ from
      $\Psi^\star = 0.95$; the dose ordering $M_\phi$ and equal-MSE blur
      penalty $B_\phi$. Cite: `viana2025perceptual`.
- [ ] Every symbol defined once; only referenced equations numbered.

### 3 Experiments and Results (500 to 650 words)

- [ ] **Dataset.** Tell: I2I 2025v1 from DeepLesion, Poisson simulation at 5
      doses, training at 100k, 899/182/220 slices, patient-level split. Cite:
      `yan2018deeplesion`, the simulation method (TODO bib).
- [ ] **Implementation.** Tell: U-Net, AdamW $10^{-4}$, batch 8, 30 epochs,
      selection on val L1, AMP, GPU, seeds, code link. Cite:
      `ronneberger2015unet`.
- [ ] **Metrics and statistics.** Tell: PSNR, SSIM, LPIPS (AlexNet), DISTS,
      $R_H$ with band definition; the circularity marks; Families A and B
      with Holm; Spearman with permutation test. Cite: `samei2019tg233`.
- [ ] **Screen paragraph (Fig 2).** Tell: which networks order doses and
      which penalize blur less than equal-MSE noise. Compare: MedSigLIP and
      CLIP vs DINOv2 vs the CT dose network, pointwise vs statistics.
- [ ] **Table 1 paragraph.** Compare: each family vs L1 on every metric with
      deltas and Holm p; say plainly which lose PSNR and which add grain
      ($R_H > 1$).
- [ ] **Fig 3 and Fig 4 paragraph.** Tell: where texture differs in the zoom
      insets; whether the screen predicts the outcome and whether size helps.
      Compare: DINOv2-S vs -B (Family B), statistics vs pointwise (Family B),
      $\rho$ with and without C2 and C3.

### 4 Conclusion (100 to 150 words)

- [ ] **C1 What we did and found.** Tell: controlled comparison; the winning
      family and matching in words.
- [ ] **C2 Recommendation.** Tell: which network and matching to use for LDCT,
      and the screen as a cheap check before training.
- [ ] **C3 Limitation and next step.** Tell: one simulated dataset, one dose
      in training, one backbone, fixed layers per network; next are real
      low-dose data and layer selection guided by the screen.

### Compliance with Ethical Standards

- [ ] "This research study was conducted retrospectively using human subject
      data made available in open access by the NIH Clinical Center
      (DeepLesion). Ethical approval was not required as confirmed by the
      license attached with the open access data."

### Acknowledgments

- [ ] Funding, grant numbers and compute resources (TODO).

### Reference Audit

- [ ] Every compared network cited where it is introduced (DINOv2, MedSigLIP,
      RadImageNet, CLIP, LPIPS, DISTS, VGG)
- [ ] DeepLesion and the Poisson simulation method cited
- [ ] TG-233 NPS cited for $R_H$
- [ ] Viana et al., ALDEN, LEDA, SACNN, D-PerceptCT and Kumar et al. cited and
      differentiated
- [ ] Metadata TODOs from novelty.md resolved (ALDEN and LEDA venues,
      RadImageNet DOI, the withdrawn Fang et al.)
- [ ] 15 to 20 references; published versions preferred over arXiv

### Figures and Tables Audit

- [ ] Fig 1 to Fig 4 and Table 1 to Table 2 referenced in order
- [ ] Captions state the takeaway
- [ ] Same display range and error scale across Fig 3 panels
- [ ] Table 1 headers carry ↑/↓ (and →1 for $R_H$), best in bold, mean ± std,
      † for Holm p < 0.05, * for circular metrics

---

## Week 5: Revisions and Submission (Mon Oct 26 to Mon Nov 2)

**Objective:** submit on Mon Nov 2, inside the expected extension, or on Mon
Oct 26 if no extension is announced.

| Day | Review milestone |
|---|---|
| Sun Oct 25 | Team comments collected |
| Mon Oct 26 | Official deadline: submit if no extension is announced; otherwise continue |
| Fri Oct 30 | All revisions merged |
| Sun Nov 1 | Final PDF checks (fonts, page limit, figures, references) |
| Mon Nov 2 | **Submit** (confirm deadline time and time zone on the ISBI site) |

- [ ] Mon Oct 26: official deadline. No extension announced: apply the
      weekend comments and submit today. Extension announced: record the new
      deadline in the README and continue.
- [ ] Fri Oct 30: all revisions merged; every comment answered or deferred
      with a reason
- [ ] Sun Nov 1: Submission Compliance below completed
- [ ] **Checkpoint (Mon Nov 2): submitted.** Post the final PDF, submission
      ID and source commit as a Markdown comment on the GitHub issue.

### Submission Compliance (verify against the ISBI 2027 call)

- [ ] Page limit for content and references respected
- [ ] Official template, margins and font sizes unchanged
- [ ] Anonymity rules followed if the call requires them
- [ ] PDF fonts embedded; IEEE PDF compliance check done if required
- [ ] Ethics section present
- [ ] Author list, order and affiliations approved by all authors
- [ ] Final PDF and LaTeX source linked in the GitHub issue
