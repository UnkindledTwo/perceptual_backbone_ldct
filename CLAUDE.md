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
  - `tables.py`: `python -m src.tables` rebuilds `table1_seed42.csv`, `table1_seed42_superseded.csv`, `table2_feature_networks.csv`, the compute CSVs and `week1_week2_tables.md` from `results/` (no GPU). It rewrites those older files, so do not run it as part of the Week 3 analysis.
  - `seeds_tables.py`: `python -m src.seeds_tables` builds everything in `results/week3/` from the 33 runs (per-slice CSV, per-seed and three-seed Table 1, statistics, validation spikes, Fig 3 slice ids, `week3_tables.md`, `provenance_seeds.json`, and since 2026-10-15 the freeze items: Table 1 with † better / ‡ worse than C0, signed deltas to C0, families per slice and per patient with seed spread, pointwise-only screen correlation marked exploratory, final Table 2, ablation of superseded runs, `disclosures.md`, `.tex` tables via `src/latex.py`). A rerun reproduces every earlier CSV byte for byte. Reuses `run_row` and `FINAL_RUNS` of `tables.py`; seed 42 folders are `train_C8_varfix`, `train_C9_v2`, `train_C10_v2`, other seeds `train_C{k}`.
  - `stats.py`: Step 3.2 statistics (Holm, rank-biserial, paired Wilcoxon families A and B, screen-versus-outcome Spearman with permutation p). The families are fixed; do not add comparisons. `unit="patient_id"` runs the same families on the 31 patient means (sensitivity check, reported next to the slice-level tests, never instead).
  - `record.py`: `guarded()` (refuses writes outside `results/week3/` and `figures/`), sha256 helpers, `results/week3/manifest.json`; `python -m src.record results/week3/results_before_week3.sha256` checks that the older result files are unchanged.
  - `val_diagnostics.py`: per-slice validation diagnostics (L1, signed error, BatchNorm on batch statistics); `python -m src.val_diagnostics DATA_ROOT train_C10_v2 ...` evaluates saved best checkpoints on CPU.
  - `model.py`: the U-Net used for all runs.
- `README.md`: public overview (three-seed Table 1, how to rebuild Week 3, layout). It was empty until 2026-10-13. Update it when results change.
- `slides_and_reports/`: presentations, CT dose network reports and the GitHub issue comments. `build_update_deck.py` builds `ldct_backbone_week_1_2_update.pdf` (corrected seed-42 slides) from `results/`; the editable source of `ldct_backboone_week_1_2.pdf` no longer exists. `LDCT Perceptual Loss Weeks 2–3-1.pdf` is the PDF export of the Weeks 2 to 3 slide deck. Issue comments (moved from the root on 2026-10-15): `week2_closeout.md` (Week 2 closeout request), `week2_closeout_reply.md`, `week3.md` (Week 3 results at `95a3842`), `week3_response.md` (supervisor review), `week3_freeze_reply.md`. Notebooks, `UNet/`, `paper_plan.md` and `README.md` stay at the root.
- `figures/fig1_overview.py`: Fig 1, final (`python -m figures.fig1_overview`, deterministic, recorded in the manifest). The plan names a `.drawio` file; the script replaces it.
- `figures/fig2_screen.py`, `figures/fig4_screen_vs_outcome.py`, `figures/style.py`: Fig 2 and Fig 4 from `results/` (`python -m figures.fig2_screen`, run from the repository root). `figures/fig3_qualitative.py` builds Fig 3 (first slice of `results/week3/fig3_slices.csv` at seeds 42, 123, 456; LDCT, C0, C1, C3, C5, C7, C8, C10, NDCT; error row at seed 42) from `results/week3/fig3/seed{S}/`, written by the `EXPORT_FIG3` cell of the seeds notebook (all three seeds, sha256 and PSNR checks). Exported on Colab from clean `a9aaa30` on 2026-10-15; the figure is built.
- `UNet/`: legacy standalone U-Net scripts. Not used by the pipeline.
- `results/`: CSV/JSON outputs. `week1_week2_tables.md` is the generated summary; `OLDRUNS/` holds superseded runs; `seed42/train_C*/` hold per-run `metrics.csv`, `train_log.csv`, `run_summary.json`; `screen/` holds per-slice screen CSVs; `provenance.json` holds the source revision, settings and checkpoint hash of every run; `val_diagnostics/` holds best-checkpoint validation diagnostics. `seed123/` and `seed456/` hold the runs of the seeds notebook (`train_C0` to `train_C10`); `week3/` holds the three-seed analysis with `manifest.json` (commands, versions, input and output hashes) and `provenance_seeds.json` (seeds 123 and 456; `provenance.json` covers seed 42 only).
- Weights (`*.pt`, `*.pth`) are gitignored, so `ct_dose_net.pt` and every `best.pth` are not in the repo. This is permanent: model weights are never pushed to GitHub, so never add them to git or change `.gitignore`. Weights move between machines through Google Drive or manual upload (Colab).

## Environment

Python 3.13, torch 2.11 (CUDA), torchvision, lpips, piq, scikit-image, scipy, pandas, matplotlib, transformers (MedSigLIP is gated on Hugging Face). Results were produced on one A100 80 GB with `cudnn.deterministic`. The data root is `minideeplesion` (Colab download in the notebook); it is not in the repo.

## Protocol (do not change without rerunning every row)

- 512×512 slices in [0, 1]. Input `noisy_clipped_100k`, target `gt_clipped`. Patient-level split 899 / 182 / 220 slices (115 / 34 / 31 patients). Test split is touched once per run.
- AdamW lr 1e-4, batch 8, 30 epochs, `ReduceLROnPlateau(factor=0.5)`, AMP. Checkpoint with the lowest validation L1. The perceptual term runs in float32.
- One shared random 224×224 crop per batch for prediction and target.
- lambda per network so the perceptual term is 95% of the initial loss (`calibrate_weight`, seed 42). Never tune lambda per network. If one config diverges, lower the target influence for all and recalibrate.
- Seeds 42, 123, 456. All three are complete (33 runs).
- 11 configurations C0 to C10 (see the plan table). DINOv2-L and CLIP are screen only.
- The screen is a prediction, not a filter: every C1 to C10 network is trained regardless of its score.

## Conventions

- Match the existing style: type hints, small functions, no heavy comments, f-strings.
- Always pass `data_range=1.0` to SSIM.
- Per-slice metrics go to CSV (`configuration`, `seed`, `slice_id`, `patient_id`, metrics). Statistics use seed-averaged per-slice values.
- When an in-text mention names a configuration ID (C8), also give the network and matching: `C8 (DINOv2-S, statistics)`.
- Do not use git unless asked. Do not delete or overwrite `results/` without being asked. New analysis outputs go to a new folder (Week 3: `results/week3/`); existing result files are never overwritten, and every generated file is recorded with its inputs and hashes in a manifest. The CPU analysis environment is `~/.venvs/ldct-analysis` (`requirements-analysis.txt`); the system Python has no numpy.

## CT dose network (C9 and C10): status and known issues

Adopted network is **v2** (log10-dose regression target, standardised, smooth L1, clean at log10 = 7, validation checkpointing; `src/dose_net_train.py`). v1 (6-way cross-entropy) is kept as a disclosed ablation. Weights: `results/seed42/ct_dose_net_v2_regression/ct_dose_net.pt` (gitignored; `results/seed42/ct_dose_net_v1|v2/` hold v1 weights). Details: `slides_and_reports/ctdosenet_v2.md`, `results/ctdosenet_followup_tables.md`.

**Status (seed 42).**
- Validation screen M (pointwise / statistics): v1 0.724 / 0.712, v2 0.971 / 0.852. B stays below 1 (0.34 / 0.86).
- The screen gain did not carry over: C9 (CT dose network v2, pointwise) PSNR 30.40, DISTS 0.097; C10 (CT dose network v2, statistics) PSNR 29.71, DISTS 0.102, best R_H of all (0.912). Both stay below C0 on PSNR and SSIM and far behind C8 (DINOv2-S, statistics) on DISTS (0.060).
- v2 was adopted under a threshold changed after seeing the screen (0.9 to 0.8, recorded in `results/decision.json`). Disclose this in the paper.
- C10 v2 validation L1 is unstable: 8 of 30 epochs above twice the median (80,138 at epoch 3, 30,478 at epoch 29), best epoch 17. Not a bug: the GPU replay (2026-10-12, `results/seed42/diag_C10_v2/`) reproduces the log exactly and shows a BatchNorm train/eval mismatch. At a spike 40 to 50 bright validation slices (17 patients) are predicted far too bright in eval mode, while BatchNorm on batch statistics shows no spike; epoch 17 has no failing slice. Spikes also occur in C2, C3, C4, C7 and C10 v1 (not replayed), never in C0. Report it in the paper as unstable eval-mode validation, not as stable training. See `slides_and_reports/c10_validation_instability.md`.
- The dose network v2 best epoch was 10 of 10, so its training was probably cut short.

**Diagnosis (CPU, 2026-10-10, 30 val slices, v2 regression weights, same MSE as the 100k noisy image).** The loss is nearly blind to structure and low-frequency error. Distance relative to white noise at equal MSE, pointwise blocks 1/2/3/4: blur 0.41/0.50/0.58/0.71, constant offset 0.16/0.25/0.32/0.35, low-frequency noise 0.16/0.23/0.26/0.25. Statistics matching: offset 0.43 to 0.61, low-frequency noise 0.39 to 0.49. Cause: receptive field about 23 px at block 3 (three conv blocks), and the network was only trained to rank noise levels on 128×128 patches. M cannot detect this (pretrained networks all score about 0.99), so the screen needs a structure term (blur, offset, low-frequency ratios) before it predicts C9/C10. Block 4 (the head's first conv block, `head[0]`, unused by the loss) gives M 0.913 pointwise and 0.843 statistics, a small gain only. Statistics block 1 alone has M 0.50. The diagnostic script is in the scratchpad, not in the repo.

**Ruled out (CPU re-screen 2026-10-08, `slides_and_reports/ctdosenet_update.md`, v1 weights):** channel normalisation (removing it hurts pointwise), crop domain gap (air-containing random crops score the same or higher), variance clamp (screen unchanged; the fix, `(variance.clamp(min=0) + eps).sqrt()`, is in `src/distances.py` and left C8 unchanged and C10 v1 worse), equal per-block weighting (no change).

**Scope decision (Week 2 closeout, 2026-10-12).** No further network variants. v2 is frozen and reported as "matches grain statistics, blind to structure"; C8 (DINOv2-S, statistics) is the practical candidate and the screen's predictive value is the scientific question. The structure-aware retrain is not pursued.

**Bookkeeping.**
- `DoseNetFeatures` ignores `prepare()`; `input_size` and the mean/std buffers are unused. Inputs are raw 1-channel crops, consistent with training but different from the plan text ("resized to the network's input size").
- Parameter counts include the classifier head; the feature extractor (three blocks) has 287,328.
- No augmentation in dose-net training.
- The Step 1.3 distance check was rerun for v2 on 2026-10-12 (`results/zoo_checks_v2.csv`): zero at equality, gradient reaches the prediction, 8.7 / 10.9 ms per batch of 8.

## Other open items

- `results/environment.json` records commit `d3916a2` (Week 1 and 2 notebook only). The revision of the follow-up runs (final C8, C9, C10) was not recorded at run time; `results/provenance.json` states what is known per run. Commit `094e416` reproduces C9 (epoch 1) and C10 (all 30 epochs, byte-identical checkpoint) exactly; C8 and C0 to C7 were not replayed.
- The Drive archives with the checkpoints (`outputs_seed42.zip`, `outputs_ctdosenet_followup.zip`) are shared as "anyone with the link can view" since 2026-10-12. The standalone v2 dose network file on Drive is still private to the owner.
- The Step 1.2 audit table of earlier runs (#208, #261) is not in the outputs.
- `RADIMAGENET_STATS` is `None` (ImageNet statistics used); confirm against the weights source before the freeze.
- The plan mentions `scripts/` and `src/unet.py`; the actual files are in `src/` (`run_config.py`, `model.py`). The "GPU hours available" cell in the compute budget is still blank.
- `src/__pycache__/` is tracked by git; ignore it.

## Three-seed results (2026-10-13, not frozen)

Source: `results/week3/week3_tables.md` (generated; seeds 42, 123, 456; ± is the standard deviation over the three per-seed means).

**Supported by the numbers.**
- Perceptual losses trade PSNR for perception. Every C1 to C10 loses PSNR against C0 (L1) (Family A, median 0.5 to 2.4 dB). The best trade-offs, C1 (VGG-19, pointwise), C2 (LPIPS-VGG, pointwise) and C8 (DINOv2-S, statistics), roughly halve LPIPS (0.104 to about 0.058) for 0.5 to 0.7 dB. This is the safest abstract number.
- Statistics matching improves grain fidelity for DINOv2-S: C8 against C5 (DINOv2-S, pointwise) is 0.195 ± 0.013 against 0.333 ± 0.044 on |1 − R_H| (Family B, r = −0.90). It is the only Family B effect clearly larger than the seed spread.
- Medical pretraining does not help. C4 (RadImageNet, pointwise) has the worst PSNR (28.70) and no LPIPS gain over C0 (Holm p = 0.09). C7 (MedSigLIP, pointwise) is smoother than C0 (R_H 0.53 against 0.66, |1 − R_H| significantly worse) at about ten times the compute.
- The CT dose network is a negative result on texture: C9 (CT dose network v2, pointwise) and C10 (CT dose network v2, statistics) stay at DISTS 0.097 to 0.098 against about 0.065 for the best group, while C10 has the best |1 − R_H| (0.167 ± 0.026). Consistent with "matches grain statistics, blind to structure".
- Seed spread is larger with a perceptual term: PSNR varies by 0.07 dB over seeds for C0 and by 0.2 to 0.5 dB for C1 to C8.

**Do not claim.**
- A ranking inside the top DISTS group: C2 0.0645 ± 0.0024, C8 0.0654 ± 0.0047, C3 (DISTS, statistics) 0.0664 ± 0.0014, C1 0.0668 ± 0.0023. The seed-42 lead of C8 does not hold, and DISTS is in neither test family.
- Small effects on the strength of their p-values. The Wilcoxon tests treat 220 slices as independent, but they come from 31 patients, and seed-averaging hides seed variance. C6 (DINOv2-B, pointwise) against C5 on PSNR is +0.05 dB with a Holm p of 2e-5 while the seed spread is 0.5 dB. C10 against C9 on |1 − R_H| (0.167 against 0.203) is about 1.4 seed spreads. Report effect sizes and seed spread next to every p-value.
- That VGG-family losses are best on texture without the circularity caveat: DISTS is VGG-based and LPIPS is an ImageNet-CNN metric, so C1 to C3 are scored partly in their own feature space (C2 on LPIPS and C3 on DISTS are marked * in Table 1).
- Mean R_H as "closest to 1". C10 has mean R_H 0.951 ± 0.058 but scatters on both sides of 1 (seed 456: 1.017); use |1 − R_H|. R_H measures the amount of high-frequency energy, not whether the texture is right.

**Screen versus outcome.** M cannot discriminate (pretrained networks all about 0.99). B against trained |1 − R_H| has rho = +0.84 (p = 0.0035, n = 10; +0.71, p = 0.059 without C2 and C3), the opposite sign to the screen's premise; B and M do not predict DISTS or LPIPS; parameter count against |1 − R_H| has rho 0.77. The p-values are uncorrected over 18 rows. Interpretation, not a tested result: B is probably confounded with the matching rule, since the three statistics configurations (C3, C8, C10) have low B and the best R_H. A hand calculation on the seven pointwise networks alone gives rho of about 0.64 (not significant at n = 7); it is post hoc and not in the result files. The statement for the paper is that the screen as designed does not predict the trained outcome.

**Other observations.**
- Validation spikes (seeds 42/123/456): C2 4/4/6, C3 2/5/3, C7 3/3/0, C10 8/0/1, C6 and C9 none. The seed spread probably ties to these spikes and to checkpoint selection on a noisy validation L1 (not tested).
- Two unexplained SSIM dips at normal PSNR: C3 at seed 42 (0.804 against 0.849 and 0.844) and C9 at seed 123 (0.827 against 0.858 and 0.869). Not investigated.
- The plan's draft Family B table listed DISTS rows, but the family is defined on PSNR, LPIPS and |1 − R_H|; the definition was kept and DISTS was not added after seeing results.

**Commit.** The Week 3 work (seeds 123 and 456, `results/week3/`, the Week 3 modules and figures, `README.md`, `paper_plan.md`, the seeds notebook) is pushed as `95a3842`. The freeze items are pushed as `a9aaa30` (2026-10-15); all Week 3 outputs were rebuilt on that clean commit and the manifest records it.

**Runs.** Seeds 123 and 456 are done (`results/seed123/` on clean `094e416`, `results/seed456/` on clean `5bed4b6`; the two commits differ only in `src/tables.py`; same lambda values and dose network sha256 as seed 42; all 22 runs pass the integrity check: 220 slices, 30 epochs, checkpoint hashes, no non-finite steps).

## Open before the freeze (Oct 18)

Supervisor review of Week 3: `slides_and_reports/week3_response.md`. Reply with per-item status and reproduce commands: `slides_and_reports/week3_freeze_reply.md` (code at `a9aaa30`, all outputs rebuilt on it; not posted yet).

1. Fig 3 is done (2026-10-15, C9 not included by decision). Optional: a zoomed region chosen by rule, because the grain differences are hard to see at full-slice size.
2. Key Result and Takeaway in `paper_plan.md`. Writing plan updates are deferred, not out of scope. The `.tex` tables compile without errors or overfull boxes at the ISBI text width (7 in, `table*`), checked on 2026-10-15 with Tectonic 0.15 (XeTeX). They are pure ASCII, so pdflatex should work too, but that was not tested. In LaTeX the configuration label is split into ID, network and matching columns (`split_configuration` in `src/seeds_tables.py`).
3. Done 2026-10-15: patient-level families (Family A 30/30, Family B 3/9 significant; C8 vs C5 on |1 − R_H| survives, C6 vs C5 on |1 − R_H| survives but flips sign at seed 123), † / ‡ marks, shared-network note and exploratory pointwise-only screen correlation, disclosures (C8 seed-42 revision closed by disclosure, no replay), final Fig 1, Table 2, LaTeX. Still optional: a look at the two SSIM dips.
4. `slides_and_reports/week2_closeout_reply.md` and `slides_and_reports/week3.md` are posted on the GitHub issue (2026-10-15; the Fig 3 commit is `66130ab`). Still to post: `week3_freeze_reply.md`, with `fig2_screen.png`, `fig3_qualitative.png` and `fig4_screen_vs_outcome.png` attached. `paper_plan.md` still points to #261.

Week 2 verification, for reference: done on the A100 from a fresh kernel and a clean clone of `094e416`: the reproduction check (`REPRO_CHECK`, relative difference 0.0), the C10 replay (`DIAGNOSE_C10`) and the v2 distance check (`CHECK_DOSE_V2_DISTANCE`). The v2 dose network is on Drive as `MyDrive/BIAI/ISBI2027_WEIGHTS/ct_dose_net_v2_regression.pt` and was accepted by sha256.

## Deliverables

Week 3 (Oct 12 to Oct 18): seeds 123 and 456 (done), Wilcoxon tests with Holm correction and screen-versus-outcome Spearman tests (done), Fig 1 to Fig 4 and the table export with LaTeX (done 2026-10-15; `.tex` tables compiled), results freeze on Oct 18. Draft Oct 23, deadline Oct 26. `slides_and_reports/ldct_backboone_week_1_2.pdf` is the Weeks 1 and 2 presentation; its result slides (6 to 12, 14, 15) are superseded by `ldct_backbone_week_1_2_update.pdf`. The CT dose network reports and follow-up notebook are in the same folder.
