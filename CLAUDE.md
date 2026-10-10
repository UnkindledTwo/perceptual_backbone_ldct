# CLAUDE.md

Research code for the ISBI 2027 benchmark paper "Which Feature Network Makes a Good Perceptual Loss for LDCT?"
(assignee Arda Şahin). A fixed U-Net is trained for low-dose CT denoising with `L1 + lambda * d_phi(crop(pred), crop(target))`,
where only the feature network `phi` and the matching rule (pointwise or local statistics) change between runs.
The full plan is `paper_plan(1).md` (protocol, steps, tables, statistics). Results live in `results/`.

## Layout

- `refactored_perceptual_backbone_ldct.ipynb`: the pipeline driver (Colab-first). Runs Steps 1.1 to 2.3 and writes `results/`. This is the source of truth for how runs were made (dose-network training, patch sampling, screen, calibration, training loop with timing).
- `perceptual_backbone_ldct.ipynb`: the old notebook. Do not reuse its numbers (SSIM anomaly, mixed protocols).
- `src/`: importable modules (`from src.x import ...`, repository root on `sys.path`).
  - `config.py`: `PerceptualConfig`, `DOSE_KEYS`.
  - `data.py`: `I2IPairs`, `load_slice`, patient-leakage check.
  - `metrics.py`: PSNR, SSIM (`data_range=1.0`), LPIPS-Alex, DISTS, high-band ratio `R_H`.
  - `feature_nets.py`: wrappers returning lists of `[B, C, h, w]` feature maps (VGG-19, RadImageNet ResNet-50, DINOv2, MedSigLIP, CLIP).
  - `ct_dose_net.py`: proposed CT dose network and `DoseNetFeatures`.
  - `distances.py`: pointwise and statistics distances, `PerceptualObjective`, `shared_crop`.
  - `screen.py`: training-free screen (dose ordering M, equal-MSE blur penalty B).
  - `calibrate_lambda.py`: lambda from a 95% perceptual share at initialisation.
  - `train.py`, `run_config.py`: reference loop and wiring. The notebook has its own timed copy of the loop; keep both consistent.
  - `model.py`: the U-Net used for all runs.
- `UNet/`: legacy standalone U-Net scripts. Not used by the pipeline.
- `results/`: CSV/JSON outputs. `week1_week2_tables.md` is the generated summary; `seed42/train_C*/` hold per-run `metrics.csv`, `train_log.csv`, `run_summary.json`; `screen/` holds per-slice screen CSVs.
- Weights (`*.pt`, `*.pth`) are gitignored, so `ct_dose_net.pt` and every `best.pth` are not in the repo. This is permanent: model weights are never pushed to GitHub, so never add them to git or change `.gitignore`. Weights move between machines through Google Drive or manual upload (Colab).

## Environment

Python 3.13, torch 2.11 (CUDA), torchvision, lpips, piq, scikit-image, scipy, pandas, matplotlib, transformers (MedSigLIP is gated on Hugging Face). Results were produced on one A100 80 GB with `cudnn.deterministic`. The data root is `minideeplesion` (Colab download in the notebook); it is not in the repo.

## Protocol (do not change without rerunning every row)

- 512×512 slices in [0, 1]. Input `noisy_clipped_100k`, target `gt_clipped`. Patient-level split 899 / 182 / 220 slices (115 / 34 / 31 patients). Test split is touched once per run.
- AdamW lr 1e-4, batch 8, 30 epochs, `ReduceLROnPlateau(factor=0.5)`, AMP. Checkpoint with the lowest validation L1. The perceptual term runs in float32.
- One shared random 224×224 crop per batch for prediction and target.
- lambda per network so the perceptual term is 95% of the initial loss (`calibrate_weight`, seed 42). Never tune lambda per network. If one config diverges, lower the target influence for all and recalibrate.
- Seeds 42, 123, 456. Only seed 42 is complete (Weeks 1 and 2).
- 11 configurations C0 to C10 (see the plan table). DINOv2-L and CLIP are screen only.
- The screen is a prediction, not a filter: every C1 to C10 network is trained regardless of its score.

## Conventions

- Match the existing style: type hints, small functions, no heavy comments, f-strings.
- Always pass `data_range=1.0` to SSIM.
- Per-slice metrics go to CSV (`configuration`, `seed`, `slice_id`, `patient_id`, metrics). Statistics use seed-averaged per-slice values.
- When an in-text mention names a configuration ID (C8), also give the network and matching: `C8 (DINOv2-S, statistics)`.
- Do not use git unless asked. Do not delete or overwrite `results/` without being asked.

## Known issues (CT dose network, C9 and C10)

Review of `src/ct_dose_net.py`, `src/distances.py`, `src/screen.py` and notebook cells 28 to 30. No crash-level bug was found: label order is consistent, BatchNorm is in eval mode when used as a loss, gradients reach the prediction, and train/val patients are disjoint. The following explain its weak results (screen ordering M about 0.72, worst DISTS among perceptual losses, best R_H). They are untested hypotheses, not confirmed causes:

1. **Channel normalisation.** `normalize_channels` divides each pixel's feature vector by its L2 norm, so only direction is compared. A noise-level classifier plausibly encodes dose in activation magnitude. Both distances apply this normalisation. Test by re-screening with unnormalised features.
2. **Crop domain gap.** The network trains on 128×128 patches with at least 90% body (mean body fraction 0.987). The loss uses random 224×224 crops from the whole 512×512 slice, including air and borders, which it never saw. The screen uses a fixed centre crop (rows and columns 144 to 368), which avoids this.
3. **Unused resize.** `DoseNetFeatures` ignores `prepare()`; `input_size` and the mean/std buffers are unused. Inputs are raw 1-channel crops at 224. This is consistent with training but differs from the plan text ("resized to the network's input size").
4. **Misleading parameter count.** Reported 1,175,142 parameters include the classifier head (about 888k). The feature extractor (three blocks) has about 287k.
5. **Shallow and thin.** Three conv blocks (the first at full resolution), 10 epochs, no validation checkpointing (last epoch is saved), no augmentation. Validation accuracy 78.5%, train accuracy about 87%.
6. **Variance clamp.** `local_stats` clamps variance at 1e-8 instead of 0, which zeroes the gradient in flat regions. Applies to every network using statistics matching.
7. **Reproducibility.** The dose-net training loop exists only in the notebook, and `ct_dose_net.pt` is gitignored. C9, C10 and the screen rows cannot be rerun without retraining.

**CPU re-screen (2026-10-08, details in `ctdosenet_update.md`):** H1, H2 and H6 do not change the screen (M stays 0.65 to 0.73; removing normalisation hurts pointwise; air-containing random crops score the same or higher; the variance clamp is irrelevant). Equal per-block weighting (v2) also changes nothing (M 0.724 to 0.726 pointwise, 0.712 statistics). The remaining suspect is H5 (small net, 10 epochs, no checkpointing or augmentation), which needs a retrain. Cheapest follow-up: per-block ablation of the screen.

## Other open items

- `results/environment.json` records commit `d3916a2`, older than the refactor that produced the results.
- The Step 1.2 audit table of earlier runs (#208, #261) is not in the outputs.
- `RADIMAGENET_STATS` is `None` (ImageNet statistics used); confirm against the weights source before the freeze.
- The plan mentions `scripts/` and `src/unet.py`; the actual files are in `src/` (`run_config.py`, `model.py`). The "GPU hours available" cell in the compute budget is still blank.
- `src/__pycache__/` is tracked by git; ignore it.
- `README.md` is empty.

## Deliverables

Week 3 (Oct 12 to Oct 18): seeds 123 and 456, Wilcoxon tests with Holm correction (families A and B), screen-versus-outcome Spearman tests, figures, results freeze on Oct 18. Draft Oct 23, deadline Oct 26. `ldct_backboone_week_1_2.pdf` is the Weeks 1 and 2 presentation.
