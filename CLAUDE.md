# CLAUDE.md

Research code for the ISBI 2027 benchmark paper "Which Feature Network Makes a Good Perceptual Loss for LDCT?"
(assignee Arda Şahin). A fixed U-Net is trained for low-dose CT denoising with `L1 + lambda * d_phi(crop(pred), crop(target))`,
where only the feature network `phi` and the matching rule (pointwise or local statistics) change between runs.
The full plan is `paper_plan.md` (protocol, steps, tables, statistics). Results live in `results/`.

## Layout

- `seeds_perceptual_backbone_ldct.ipynb`: **the only entry point for new runs** (seeds 123 and 456). Trains C0 to C10 from the frozen lambda values and the frozen regression-v2 dose network (chosen by sha256), records commit, dirty flag and source hashes in every `run_summary.json`. Optional sections: reproduction check against seed 42, Step 1.3 check of the v2 dose network, C10 replay with validation diagnostics.
- `refactored_perceptual_backbone_ldct.ipynb`: record of the seed-42 Weeks 1 and 2 runs (Steps 1.1 to 2.3, C0 to C7, pre-fix C8, v1 C9/C10). Its dose-network cells are the superseded v1 setup. Not a runner for new seeds. `slides_and_reports/ctdosenet_followup_ldct.ipynb` is the record of the final C8, C9 and C10 runs.
- `perceptual_backbone_ldct.ipynb`: the old notebook. Do not reuse its numbers (SSIM anomaly, mixed protocols).
- `src/`: importable modules (`from src.x import ...`, repository root on `sys.path`).
  - `config.py`: `PerceptualConfig`, `DOSE_KEYS`.
  - `data.py`: `I2IPairs`, `load_slice`, patient-leakage check.
  - `metrics.py`: PSNR, SSIM (`data_range=1.0`), LPIPS-Alex, DISTS, high-band ratio `R_H`.
  - `feature_nets.py`: wrappers returning lists of `[B, C, h, w]` feature maps (VGG-19, RadImageNet ResNet-50, DINOv2, MedSigLIP, CLIP).
  - `ct_dose_net.py`: proposed CT dose network and `DoseNetFeatures`.
  - `dose_net_train.py`: dose-network training (body-patch sampling, loss modes, `train_dose_net`).
  - `distances.py`: pointwise and statistics distances, `PerceptualObjective`, `shared_crop`.
  - `screen.py`: training-free screen (dose ordering M, equal-MSE blur penalty B).
  - `calibrate_lambda.py`: lambda from a 95% perceptual share at initialisation.
  - `train.py`, `run_config.py`: reference loop and wiring (`load_dose_net` reads the head width from the checkpoint, so v1 and v2 both load). The notebooks have their own timed copy of the loop; keep them consistent.
  - `tables.py`: `python -m src.tables` rebuilds `table1_seed42.csv`, `table1_seed42_superseded.csv`, `table2_feature_networks.csv`, the compute CSVs and `week1_week2_tables.md` from `results/` (no GPU).
  - `val_diagnostics.py`: per-slice validation diagnostics (L1, signed error, BatchNorm on batch statistics); `python -m src.val_diagnostics DATA_ROOT train_C10_v2 ...` evaluates saved best checkpoints on CPU.
  - `model.py`: the U-Net used for all runs.
- `README.md`: public overview (configurations, protocol, layout, seed-42 results). Update it when results change.
- `slides_and_reports/`: presentation and CT dose network reports. `build_update_deck.py` builds `ldct_backbone_week_1_2_update.pdf` (corrected seed-42 slides) from `results/`; the editable source of `ldct_backboone_week_1_2.pdf` no longer exists.
- `figures/fig1_overview.py`: Fig 1 draft (writes `.svg`, `.pdf`, `.png`). The plan names a `.drawio` file; the script replaces it.
- `UNet/`: legacy standalone U-Net scripts. Not used by the pipeline.
- `results/`: CSV/JSON outputs. `week1_week2_tables.md` is the generated summary; `OLDRUNS/` holds superseded runs; `seed42/train_C*/` hold per-run `metrics.csv`, `train_log.csv`, `run_summary.json`; `screen/` holds per-slice screen CSVs; `provenance.json` holds the source revision, settings and checkpoint hash of every run; `val_diagnostics/` holds best-checkpoint validation diagnostics.
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

## CT dose network (C9 and C10): status and known issues

Adopted network is **v2** (log10-dose regression target, standardised, smooth L1, clean at log10 = 7, validation checkpointing; `src/dose_net_train.py`). v1 (6-way cross-entropy) is kept as a disclosed ablation. Weights: `results/seed42/ct_dose_net_v2_regression/ct_dose_net.pt` (gitignored; `results/seed42/ct_dose_net_v1|v2/` hold v1 weights). Details: `slides_and_reports/ctdosenet_v2.md`, `results/ctdosenet_followup_tables.md`.

**Status (seed 42).**
- Validation screen M (pointwise / statistics): v1 0.724 / 0.712, v2 0.971 / 0.852. B stays below 1 (0.34 / 0.86).
- The screen gain did not carry over: C9 (CT dose network v2, pointwise) PSNR 30.40, DISTS 0.097; C10 (CT dose network v2, statistics) PSNR 29.71, DISTS 0.102, best R_H of all (0.912). Both stay below C0 on PSNR and SSIM and far behind C8 (DINOv2-S, statistics) on DISTS (0.060).
- v2 was adopted under a threshold changed after seeing the screen (0.9 to 0.8, recorded in `results/decision.json`). Disclose this in the paper.
- C10 v2 validation L1 is unstable: 8 of 30 epochs above twice the median (80,138 at epoch 3, 30,478 at epoch 29), best epoch 17. No bug found (checkpoint reproduces the logged value on CPU; at its best epoch it looks like C0), cause open until the GPU replay (`DIAGNOSE_C10` in the seeds notebook) is run. Spikes also occur in C2, C3, C4, C7 and C10 v1, never in C0. See `slides_and_reports/c10_validation_instability.md`.
- The dose network v2 best epoch was 10 of 10, so its training was probably cut short.

**Diagnosis (CPU, 2026-10-10, 30 val slices, v2 regression weights, same MSE as the 100k noisy image).** The loss is nearly blind to structure and low-frequency error. Distance relative to white noise at equal MSE, pointwise blocks 1/2/3/4: blur 0.41/0.50/0.58/0.71, constant offset 0.16/0.25/0.32/0.35, low-frequency noise 0.16/0.23/0.26/0.25. Statistics matching: offset 0.43 to 0.61, low-frequency noise 0.39 to 0.49. Cause: receptive field about 23 px at block 3 (three conv blocks), and the network was only trained to rank noise levels on 128×128 patches. M cannot detect this (pretrained networks all score about 0.99), so the screen needs a structure term (blur, offset, low-frequency ratios) before it predicts C9/C10. Block 4 (the head's first conv block, `head[0]`, unused by the loss) gives M 0.913 pointwise and 0.843 statistics, a small gain only. Statistics block 1 alone has M 0.50. The diagnostic script is in the scratchpad, not in the repo.

**Ruled out (CPU re-screen 2026-10-08, `slides_and_reports/ctdosenet_update.md`, v1 weights):** channel normalisation (removing it hurts pointwise), crop domain gap (air-containing random crops score the same or higher), variance clamp (screen unchanged; the fix, `(variance.clamp(min=0) + eps).sqrt()`, is in `src/distances.py` and left C8 unchanged and C10 v1 worse), equal per-block weighting (no change).

**Scope decision (Week 2 closeout, 2026-10-12).** No further network variants. v2 is frozen and reported as "matches grain statistics, blind to structure"; C8 (DINOv2-S, statistics) is the practical candidate and the screen's predictive value is the scientific question. The structure-aware retrain is not pursued.

**Bookkeeping.**
- `DoseNetFeatures` ignores `prepare()`; `input_size` and the mean/std buffers are unused. Inputs are raw 1-channel crops, consistent with training but different from the plan text ("resized to the network's input size").
- Parameter counts include the classifier head; the feature extractor (three blocks) has 287,328.
- No augmentation in dose-net training.
- The Step 1.3 distance check was not rerun for v2.

## Other open items

- `results/environment.json` records commit `d3916a2` (Week 1 and 2 notebook only). The revision of the follow-up runs (final C8, C9, C10) was not recorded at run time; `results/provenance.json` states what is known per run.
- The Drive archives with the checkpoints (`outputs_seed42.zip`, `outputs_ctdosenet_followup.zip`) are private to the owner's account; the v2 dose network has no standalone Drive file.
- Not yet run on the GPU: the fresh-kernel reproduction check of the seeds notebook (`REPRO_CHECK`), the C10 replay (`DIAGNOSE_C10`) and the Step 1.3 check for v2 (`CHECK_DOSE_V2_DISTANCE`).
- The Step 1.2 audit table of earlier runs (#208, #261) is not in the outputs.
- `RADIMAGENET_STATS` is `None` (ImageNet statistics used); confirm against the weights source before the freeze.
- The plan mentions `scripts/` and `src/unet.py`; the actual files are in `src/` (`run_config.py`, `model.py`). The "GPU hours available" cell in the compute budget is still blank.
- `src/__pycache__/` is tracked by git; ignore it.

## Deliverables

Week 3 (Oct 12 to Oct 18): seeds 123 and 456, Wilcoxon tests with Holm correction (families A and B), screen-versus-outcome Spearman tests, figures, results freeze on Oct 18. Draft Oct 23, deadline Oct 26. `slides_and_reports/ldct_backboone_week_1_2.pdf` is the Weeks 1 and 2 presentation; the CT dose network reports and follow-up notebook are in the same folder.
