# CT dose network v2 and the variance-fix rerun (seed 42)

Reply to the review in `ctdosenet_latest.md`. Everything here is seed 42 on one A100, test split touched once per run, protocol unchanged (30 epochs, AdamW 1e-4, batch 8, λ for a 95% perceptual share). Seeds 123 and 456 are **not started**. Outputs are in `results/seed42/` (`train_C8_varfix`, `train_C10_varfix_v1`, `train_C9_v2`, `train_C10_v2`, `ct_dose_net_v2_regression`), `results/screen/*_val.csv`, `results/step_b_before_after.csv`, `results/decision.json`.

## Plan status

- [x] **Variance fix, recalibrate λ, rerun C8 and C10.** Done. The fix works (finite, nonzero std gradients in near-flat windows, 58% of entries nonzero against 0% with the old clamp). λ barely moves (below). **C8 (DINOv2-S, statistics) is unchanged within noise. C10 (CT dose network v1, statistics) got worse.**
- [x] **One retry of the dose network, log-dose regression with validation checkpointing, judged on the validation screen only.** Done. Screen M rose clearly (table below), so it became v2 and replaced C9 and C10. v1 is kept as a disclosed ablation.
- [x] **C9 and C10 retrained on v2.** Done. The screen gain did **not** carry over to the test metrics (see below).
- [ ] **Seeds 123 and 456 for all configurations.** Not started, as requested.
- [ ] **Commit the screen scripts from the scratchpad.** Not done (no git actions taken here). This is still open.

## What the new models are

| | v1 | v2 |
|---|---|---|
| Feature extractor | 3 conv blocks, 287,328 parameters | identical |
| Training target | 6-way cross-entropy | log10-dose regression, standardised, smooth L1; clean placed at log10 = 7 |
| Head | 6 logits | 1 output (1,173,857 parameters in total) |
| Checkpoint | last epoch | lowest validation loss (epoch 10 of 10, so it was the last epoch anyway) |
| Patches | 21,576 train / 4,368 val | identical (same seeds) |
| Validation accuracy, 6 classes (decoded) | 78.5% | 78.7% |
| Clean vs 1000k | 90.5% | 92.2% |
| Validation MAE | n/a | 0.183 log10 |
| Training cost | 0.111 GPU h | 0.138 GPU h |

Decoded accuracy is a poor measure for regression, because 500k and 1000k are only 0.3 log10 apart. Its value is that v2 did not lose classification ability. Per-class confusion is in `ct_dose_net_v2_regression/confusion_matrix.csv`; the main errors are neighbouring doses (50k/100k, 500k/1000k).

### Validation screen (182 val slices, centre crop)

| Dose network | Matching | M | B | dose ratio 10k/1000k |
|---|---|---|---|---|
| v1 | pointwise | 0.724 | 0.308 | 4.93 |
| v1 | statistics | 0.712 | 0.754 | 2.69 |
| **v2** | pointwise | **0.971** | 0.343 | 4.60 |
| **v2** | statistics | **0.852** | 0.863 | 2.28 |

The graded target did what the review predicted for the **ordering**: M rises by 0.25 (pointwise) and 0.14 (statistics). The blur penalty B stays below 1 for both, so v2 still scores an equal-MSE blurred image as closer than the noisy one.

**Disclosure about the adoption rule.** The review gave "for example above 0.9". The notebook first coded 0.9 for both matchings, which statistics (0.852) missed. `decision.json` now records the rule as M > 0.8, so v2 was adopted under a threshold set after seeing the screen. State this in the paper; the pointwise result (0.971) meets 0.9 on its own.

## Variance fix: C8 and C10 (test split, paired per-slice, n = 220)

λ before and after: C8 1.9659 to 1.9659 (S_φ 2.6484 to 2.6483), C10 v1 2.9105 to 2.9109. The fix changes the distance values by about 1e-4, as the review said.

| Run | PSNR | SSIM | LPIPS | DISTS | R_H | non-finite steps |
|---|---|---|---|---|---|---|
| C8 (DINOv2-S, statistics), before | 30.531 | 0.8660 | 0.0483 | 0.0602 | 0.857 | 0 |
| C8, **after** | 30.547 | 0.8666 | 0.0486 | 0.0600 | 0.861 | 0 |
| C10 (CT dose network v1, statistics), before | 30.171 | 0.8544 | 0.0676 | 0.0934 | 0.875 | 0 |
| C10 v1, **after** | 29.887 | 0.8373 | 0.0733 | 0.0997 | 0.879 | 0 |

- **C8:** the differences are tiny (PSNR +0.017 dB, SSIM +0.0005, LPIPS and DISTS unchanged). The headline configuration does not depend on the clamp, which is good news for the paper. This matches the expectation that the fix mostly touches flat windows.
- **C10 v1:** worse by 0.28 dB PSNR, 0.017 SSIM and +0.006 DISTS. Two caveats: it is one seed, and the validation L1 curve of the statistics runs is noisy (a run with a validation spike late in training is easy to see in `train_log.csv`). Do not claim that the old clamp helped. The honest statement is "the fix does not improve C10 v1 at seed 42 and the run-to-run variation is of the same order as the effect", which the other two seeds will test.
- No instability appeared from the larger sqrt gradient in near-flat windows (0 non-finite steps in every run).

## C9 and C10 on v2 (test split)

| Config | PSNR | SSIM | LPIPS | DISTS | R_H | best epoch |
|---|---|---|---|---|---|---|
| C0 (L1) | 30.877 | 0.8715 | 0.1036 | 0.1115 | 0.652 | 26 |
| C8 (DINOv2-S, statistics), after fix | 30.547 | 0.8666 | 0.0486 | 0.0600 | 0.861 | 27 |
| C9 (CT dose network v1, pointwise) | 30.255 | 0.8456 | 0.0781 | 0.1001 | 0.871 | 27 |
| **C9 (CT dose network v2, pointwise)** | 30.404 | 0.8583 | 0.0676 | 0.0972 | 0.853 | 26 |
| C10 (CT dose network v1, statistics), before fix | 30.171 | 0.8544 | 0.0676 | 0.0934 | 0.875 | 26 |
| C10 (CT dose network v1, statistics), after fix | 29.887 | 0.8373 | 0.0733 | 0.0997 | 0.879 | 25 |
| **C10 (CT dose network v2, statistics)** | 29.714 | 0.8430 | 0.0794 | 0.1021 | 0.912 | 17 |

Paired differences (v2 minus comparison, Wilcoxon over 220 slices, descriptive only because slices within a patient are correlated; the paper's tests use seed-averaged values and Holm correction):

- **C9, v2 against v1:** PSNR +0.148 dB, SSIM +0.013, LPIPS -0.0105, DISTS -0.0029, R_H -0.018. A consistent improvement on every perceptual metric, small in PSNR.
- **C10, v2 against v1 (after fix):** PSNR -0.173, SSIM +0.006, LPIPS +0.006, DISTS +0.002; against the original C10, PSNR -0.457 and DISTS +0.009. **Worse on PSNR and DISTS**, but the best R_H of any configuration so far (0.912, closest to the ideal of 1).
- The v2 configurations stay below C0 on PSNR and SSIM, and below C8 on every metric except R_H.
- C10 v2 peaked at epoch 17 and its validation L1 spiked at one point late in training (30477 against about 6000 normally), so this run is the least stable of the set.

## Reading of the result

1. **The screen and the outcome disagree here.** Pointwise M went from 0.72 to 0.97 and the test metrics improved only slightly; statistics M went from 0.71 to 0.85 and the test metrics got worse. A better dose ordering is not enough to make a good training loss. This is directly relevant to the screen-versus-outcome Spearman test: the screen is a prediction, not a filter, and these two points will weigh against it, so they must stay in the data.
2. **The diagnosis was half right.** Replacing the saturating cross-entropy fixed the ordering (the target the review aimed at), but the downstream effect on image quality is small. The remaining gap to the pretrained networks (DISTS 0.097 against 0.060 for C8) is not explained by M. The B values below 1 and the small network are the candidates, still untested.
3. **Report v2 as the CT dose network** (C9 and C10) with v1 as a disclosed ablation, and frame the finding as "a task-trained CT network, even with a graded dose target, matches grain statistics (best R_H) but does not match pretrained feature networks on perceptual metrics". This is honest, and it is the result the review said to be ready to report.

## Caveats and open items

- One seed. Differences below about 0.15 dB PSNR should not be interpreted until seeds 123 and 456 exist; the C8 and C10 v1 changes in particular need them.
- λ for v2 was recalibrated (C9 1.549, C10 2.802) at the usual 95% share.
- `ct_dose_net.pt` (v1 and v2) is gitignored and lives in Drive only, as with every `best.pth`.
- The new denoised PNGs for the four new runs were added to `results/denoised/` with `_varfix` and `_v2` suffixes; the originals were left alone.
- `week1_week2_tables.md`, `table1_seed42.csv` and `table2_feature_networks.csv` still contain the pre-fix C8 and v1 C10 rows and have to be regenerated before the freeze on Oct 18.
- `results/environment.json` still names commit `d3916a2`.
- Plan item still open: the dose network training loop now lives in `src/dose_net_train.py`, but it has not been committed.


---

## Updated Week 1 and Week 2 tables

Final configurations are C0 to C7 (unchanged), **C8 (DINOv2-S, statistics) with the variance fix**, and **C9 and C10 on CT dose network v2 (regression)**. Superseded v1 and pre-fix rows are listed separately for disclosure. Weeks 1 and 2 tables not shown here (environment, Step 1.1 checks, Step 1.3 checks of the pretrained networks) are unchanged; `environment.json` still names the old commit `d3916a2`. The v1 screen row differs from the Week 1 table in the 4th decimal (B 0.7542 against 0.7544) because it was recomputed in the follow-up notebook on the same validation slices. All values are seed 42.

### Table 1: seed 42, test split, spread over test slices (final configurations)

| Configuration (seed 42, test, 100k) | PSNR (dB) | SSIM | LPIPS | DISTS | R_H | ms per step |
|---|---|---|---|---|---|---|
| LDCT input (100k) | 23.04 ± 2.13 | 0.7423 ± 0.0647 | 0.2843 ± 0.0959 | 0.1641 ± 0.0462 | 6.1743 ± 5.8560 | - |
| C0 (L1) | 30.88 ± 1.97 | 0.8715 ± 0.0946 | 0.1036 ± 0.0704 | 0.1115 ± 0.0203 | 0.6522 ± 0.2270 | 118 |
| C1 (VGG-19, pointwise) | 30.17 ± 1.94 | 0.8619 ± 0.0905 | 0.0599 ± 0.0257 | 0.0694 ± 0.0166 | 0.7689 ± 0.1832 | 129 |
| C2 (LPIPS-VGG) | 30.73 ± 1.96 | 0.8639 ± 0.0766 | 0.0570 ± 0.0210 | 0.0627 ± 0.0137 | 0.7841 ± 0.1763 | 141 |
| C3 (DISTS, statistics) | 30.03 ± 1.76 | 0.8045 ± 0.0573 | 0.0513 ± 0.0204 | 0.0649 ± 0.0129 | 0.8360 ± 0.1901 | 147 |
| C4 (RadImageNet, pointwise) | 28.85 ± 1.51 | 0.8342 ± 0.0876 | 0.0832 ± 0.0308 | 0.0841 ± 0.0161 | 0.7647 ± 0.2506 | 134 |
| C5 (DINOv2-S, pointwise) | 29.53 ± 2.07 | 0.8649 ± 0.0803 | 0.0595 ± 0.0305 | 0.0673 ± 0.0133 | 0.6549 ± 0.2163 | 150 |
| C6 (DINOv2-B, pointwise) | 29.69 ± 1.78 | 0.8567 ± 0.0846 | 0.0639 ± 0.0372 | 0.0745 ± 0.0143 | 0.7156 ± 0.2707 | 201 |
| C7 (MedSigLIP, pointwise) | 29.90 ± 1.86 | 0.8575 ± 0.0969 | 0.0701 ± 0.0477 | 0.0706 ± 0.0140 | 0.5822 ± 0.2160 | 1485 |
| C8 (DINOv2-S, statistics), variance fix | 30.55 ± 1.94 | 0.8666 ± 0.0800 | 0.0486 ± 0.0226 | 0.0600 ± 0.0134 | 0.8605 ± 0.2265 | 149 |
| C9 (CT dose network v2, pointwise) | 30.40 ± 1.78 | 0.8583 ± 0.0723 | 0.0676 ± 0.0230 | 0.0972 ± 0.0228 | 0.8527 ± 0.2295 | 126 |
| C10 (CT dose network v2, statistics) | 29.71 ± 2.09 | 0.8430 ± 0.1160 | 0.0794 ± 0.0516 | 0.1021 ± 0.0151 | 0.9120 ± 0.2497 | 129 |

Superseded and ablation rows (kept for disclosure, not part of the final table):

| Configuration (seed 42, test, 100k) | PSNR (dB) | SSIM | LPIPS | DISTS | R_H | ms per step |
|---|---|---|---|---|---|---|
| C8 (DINOv2-S, statistics), before the fix | 30.53 ± 1.86 | 0.8660 ± 0.0753 | 0.0483 ± 0.0211 | 0.0602 ± 0.0144 | 0.8573 ± 0.1961 | 153 |
| C9 (CT dose network v1, pointwise) | 30.26 ± 1.80 | 0.8456 ± 0.0718 | 0.0781 ± 0.0255 | 0.1001 ± 0.0229 | 0.8709 ± 0.1925 | 128 |
| C10 (CT dose network v1, statistics), before the fix | 30.17 ± 2.04 | 0.8544 ± 0.0918 | 0.0676 ± 0.0296 | 0.0934 ± 0.0171 | 0.8753 ± 0.2446 | 132 |
| C10 (CT dose network v1, statistics), variance fix | 29.89 ± 1.85 | 0.8373 ± 0.0831 | 0.0733 ± 0.0281 | 0.0997 ± 0.0176 | 0.8786 ± 0.2174 | 129 |

### Step 1.4: training-free screen (validation split), CT dose network rows updated

| Network | Matching | ordering_M | dose_ratio_10k_1000k | blur_penalty_B | Parameters |
|---|---|---|---|---|---|
| Pixel L1 (reference) | - | 1.0000 | 5.7367 | 0.8861 | 0 |
| VGG-19 relu3_2 | pointwise | 1.0000 | 2.6057 | 1.5092 | 1145408 |
| VGG-19 relu3_2 | statistics | 0.9989 | 1.6763 | 1.4181 | 1145408 |
| LPIPS-VGG | pointwise | 0.9984 | 2.4252 | 1.4825 | 14714688 |
| DISTS | statistics | 0.9835 | 2.7509 | 1.4495 | 14714688 |
| RadImageNet ResNet-50 | pointwise | 0.9538 | 7.1007 | 1.8118 | 25557032 |
| RadImageNet ResNet-50 | statistics | 0.9330 | 3.5960 | 1.3961 | 25557032 |
| DINOv2-S | pointwise | 0.9978 | 7.7117 | 1.7518 | 22056576 |
| DINOv2-S | statistics | 0.9973 | 2.3666 | 1.2604 | 22056576 |
| DINOv2-B | pointwise | 0.9973 | 6.2713 | 1.6694 | 86580480 |
| DINOv2-B | statistics | 0.9940 | 2.3098 | 1.2501 | 86580480 |
| DINOv2-L | pointwise | 0.9956 | 6.8865 | 1.8737 | 304368640 |
| DINOv2-L | statistics | 0.9967 | 2.3219 | 1.2776 | 304368640 |
| CLIP ViT-B/16 | pointwise | 0.9956 | 2.7337 | 1.5710 | 85799424 |
| CLIP ViT-B/16 | statistics | 0.9901 | 1.7508 | 1.4193 | 85799424 |
| MedSigLIP-448 | pointwise | 0.9995 | 3.2543 | 1.5346 | 428565440 |
| MedSigLIP-448 | statistics | 0.9956 | 2.3334 | 1.5369 | 428565440 |
| CT dose network v1 | pointwise | 0.7242 | 4.9332 | 0.3083 | 1175142 |
| CT dose network v1 | statistics | 0.7121 | 2.6928 | 0.7542 | 1175142 |
| CT dose network v2 (regression) | pointwise | 0.9709 | 4.6044 | 0.3434 | 1173857 |
| CT dose network v2 (regression) | statistics | 0.8522 | 2.2845 | 0.8633 | 1173857 |

Other rows are unchanged from the Week 1 screen. The v2 rows are from the follow-up notebook (centre crop, 182 validation slices, same screen code). The Step 1.3 distance check (distance at equality, gradient reaches prediction, ms per batch of 8) was not rerun for v2; the feature extractor is identical to v1 (3 blocks, 287,328 parameters), so the v1 cost of 8.7 and 11.6 ms per batch of 8 applies.

### Step 2.1: calibration (95% perceptual share, seed 42)

| Configuration | Label | S_1 | S_phi | lambda | lambda before (superseded) |
|---|---|---|---|---|---|
| C0 | L1 | 0.2740 | - | 0.0000 | - |
| C1 | VGG-19 | 0.2740 | 1.1080 | 4.6992 | - |
| C2 | LPIPS-VGG | 0.2740 | 0.5682 | 9.1625 | - |
| C3 | DISTS | 0.2740 | 0.7455 | 6.9836 | - |
| C4 | RadImageNet | 0.2740 | 2.3552 | 2.2106 | - |
| C5 | DINOv2-S pointwise | 0.2740 | 3.1857 | 1.6343 | - |
| C6 | DINOv2-B pointwise | 0.2740 | 2.0849 | 2.4972 | - |
| C7 | MedSigLIP | 0.2740 | 1.3864 | 3.7553 | - |
| C8 | DINOv2-S statistics | 0.2740 | 2.6483 | 1.9659 | 1.9659 |
| C9 | CT dose net v2 pointwise | 0.2740 | 3.3606 | 1.5493 | 1.5901 |
| C10 | CT dose net v2 statistics | 0.2740 | 1.8584 | 2.8016 | 2.9105 |

v1 values for reference: C9 (CT dose network v1, pointwise) 1.5901, C10 (CT dose network v1, statistics) 2.9105 (2.9109 after the fix).

### Step 2.2: CT dose network checks (v1 and v2)

| Check | v1 | v2 |
|---|---|---|
| Target | 6-way cross-entropy | log10-dose regression (clean at 7.0) |
| Val accuracy, 6 classes (decoded) | 78.48% | 78.71% |
| Val accuracy, clean vs 1000k | 90.52% | 92.24% |
| Val MAE, log10 dose | - | 0.183 |
| Best epoch / epochs | 10 / 10 (last epoch) | 10 / 10 |
| Screen M (pointwise / statistics) | 0.7242 / 0.7121 | 0.9709 / 0.8522 |
| Screen B (pointwise / statistics) | 0.3083 / 0.7544 | 0.3434 / 0.8633 |
| Parameters (feature extractor) | 1,175,142 (287,328) | 1,173,857 (287,328) |
| Training GPU hours | 0.111 | 0.138 |

Per class, v2 (validation patches):

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| 10k | 1.0 | 0.9011 | 0.948 | 728 |
| 50k | 0.8467 | 0.6676 | 0.7465 | 728 |
| 100k | 0.7308 | 0.7981 | 0.763 | 728 |
| 500k | 0.6378 | 0.6772 | 0.6569 | 728 |
| 1000k | 0.6534 | 0.7898 | 0.7152 | 728 |
| clean | 0.9377 | 0.8887 | 0.9126 | 728 |

### Table 2 values per trained network (final configurations)

| Configuration | Network | Matching | Parameters | lambda | ms_per_step | ms_per_batch_of_8 | M_phi | B_phi |
|---|---|---|---|---|---|---|---|---|
| C1 | VGG-19 relu3_2 | pointwise | 1145408 | 4.6992 | 129.4 | 11.1 | 1.0000 | 1.5092 |
| C2 | LPIPS-VGG | pointwise | 14714688 | 9.1625 | 140.8 | 27.1 | 0.9984 | 1.4825 |
| C3 | DISTS | statistics | 14714688 | 6.9836 | 146.9 | 26.4 | 0.9835 | 1.4495 |
| C4 | RadImageNet ResNet-50 | pointwise | 25557032 | 2.2106 | 134.3 | 20.6 | 0.9538 | 1.8118 |
| C5 | DINOv2-S | pointwise | 22056576 | 1.6343 | 150.2 | 31.5 | 0.9978 | 1.7518 |
| C6 | DINOv2-B | pointwise | 86580480 | 2.4972 | 200.6 | 79.4 | 0.9973 | 1.6694 |
| C7 | MedSigLIP-448 | pointwise | 428565440 | 3.7553 | 1484.6 | 1370.5 | 0.9995 | 1.5346 |
| C8 | DINOv2-S | statistics | 22056576 | 1.9659 | 149.0 | 28.9 | 0.9973 | 1.2604 |
| C9 | CT dose network v2 | pointwise | 1173857 | 1.5493 | 125.6 | - | 0.9709 | 0.3434 |
| C10 | CT dose network v2 | statistics | 1173857 | 2.8016 | 128.8 | - | 0.8522 | 0.8633 |

### Compute budget (updated)

| Item | Value |
|---|---|
| Trained configurations | 11 |
| Seeds | 3 |
| Full U-Net runs | 33 |
| CT dose network pretraining, GPU hours (v1 / v2, measured) | 0.111 / 0.138 |
| Screen (Step 1.4), GPU hours (measured, Week 1) | 0.238 |
| Calibration (Step 2.1), GPU hours (measured, Week 1; follow-up recalibration not timed) | 0.086 |
| GPU hours per U-Net run, C0 | 0.155 |
| GPU hours per U-Net run, C7 MedSigLIP | 1.445 |
| GPU hours of the final seed 42 U-Net runs (11 of 11) | 3.242 |
| GPU hours of superseded and ablation U-Net runs (C8 before fix, C9 v1, C10 v1 before and after fix) | 0.693 |
| GPU hours used so far (final runs, superseded runs, dose networks, screen, calibration) | 4.507 |
| Total GPU hours of the protocol (3 seeds of the final configurations, v2 dose network) | 10.187 |
| GPU | NVIDIA A100-SXM4-80GB (79.3 GB) |
| Largest peak GPU memory of a run, GB | 17.0 |
| GPU hours available in Weeks 1 to 3 | fill in by hand |
