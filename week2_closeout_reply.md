## Week 2 closeout: reply

Draft of the issue comment, 2026-10-12. Everything below is seed 42. The changes are in the working tree and are **not committed or pushed yet**; the paths resolve once they are.

### Key result and takeaway

**Key result.** Perceptual terms lower LPIPS by 20 to 53% and DISTS by 8 to 46% against C0 (L1) at a cost of 0.15 to 2.0 dB. C8 (DINOv2-S, statistics) is the best trade-off: 30.55 dB (0.33 dB below C0), LPIPS 0.0486, DISTS 0.0600, R_H 0.860.

**Takeaway.** The CT dose network v2 raised the screen ordering M from 0.72 to 0.97 (pointwise) and from 0.71 to 0.85 (statistics), but C9 (CT dose network v2, pointwise) and C10 (CT dose network v2, statistics) did not improve uniformly and keep the worst DISTS of the perceptual losses (0.097, 0.102). M alone does not predict the trained outcome. Whether the screen predicts it at all is the Week 3 question; C8 is the practical candidate.

### Table 1, final configurations (test split, 220 slices)

| Configuration | PSNR (dB) | SSIM | LPIPS | DISTS | R_H | ms per step | best epoch |
|---|---|---|---|---|---|---|---|
| LDCT input (100k) | 23.04 ± 2.13 | 0.7423 ± 0.0647 | 0.2843 ± 0.0959 | 0.1641 ± 0.0462 | 6.1743 ± 5.8560 | - | - |
| C0 (L1) | 30.88 ± 1.97 | 0.8715 ± 0.0946 | 0.1036 ± 0.0704 | 0.1115 ± 0.0203 | 0.6522 ± 0.2270 | 118 | 26 |
| C1 (VGG-19, pointwise) | 30.17 ± 1.94 | 0.8619 ± 0.0905 | 0.0599 ± 0.0257 | 0.0694 ± 0.0166 | 0.7689 ± 0.1832 | 129 | 27 |
| C2 (LPIPS-VGG, pointwise) | 30.73 ± 1.96 | 0.8639 ± 0.0766 | 0.0570 ± 0.0210 | 0.0627 ± 0.0137 | 0.7841 ± 0.1763 | 141 | 26 |
| C3 (DISTS, statistics) | 30.03 ± 1.76 | 0.8045 ± 0.0573 | 0.0513 ± 0.0204 | 0.0649 ± 0.0129 | 0.8360 ± 0.1901 | 147 | 28 |
| C4 (RadImageNet, pointwise) | 28.85 ± 1.51 | 0.8342 ± 0.0876 | 0.0832 ± 0.0308 | 0.0841 ± 0.0161 | 0.7647 ± 0.2506 | 134 | 27 |
| C5 (DINOv2-S, pointwise) | 29.53 ± 2.07 | 0.8649 ± 0.0803 | 0.0595 ± 0.0305 | 0.0673 ± 0.0133 | 0.6549 ± 0.2163 | 150 | 23 |
| C6 (DINOv2-B, pointwise) | 29.69 ± 1.78 | 0.8567 ± 0.0846 | 0.0639 ± 0.0372 | 0.0745 ± 0.0143 | 0.7156 ± 0.2707 | 201 | 19 |
| C7 (MedSigLIP, pointwise) | 29.90 ± 1.86 | 0.8575 ± 0.0969 | 0.0701 ± 0.0477 | 0.0706 ± 0.0140 | 0.5822 ± 0.2160 | 1485 | 22 |
| C8 (DINOv2-S, statistics), variance fix | 30.55 ± 1.94 | 0.8666 ± 0.0800 | 0.0486 ± 0.0226 | 0.0600 ± 0.0134 | 0.8605 ± 0.2265 | 149 | 27 |
| C9 (CT dose network v2, pointwise) | 30.40 ± 1.78 | 0.8583 ± 0.0723 | 0.0676 ± 0.0230 | 0.0972 ± 0.0228 | 0.8527 ± 0.2295 | 126 | 26 |
| C10 (CT dose network v2, statistics) | 29.71 ± 2.09 | 0.8430 ± 0.1160 | 0.0794 ± 0.0516 | 0.1021 ± 0.0151 | 0.9120 ± 0.2497 | 129 | 17 |

± is the standard deviation over the 220 test slices of one seed, not a spread over seeds.

### Superseded and ablation runs (disclosure only)

| Configuration | PSNR (dB) | SSIM | LPIPS | DISTS | R_H | best epoch |
|---|---|---|---|---|---|---|
| C8 (DINOv2-S, statistics), before the variance fix | 30.53 ± 1.86 | 0.8660 ± 0.0753 | 0.0483 ± 0.0211 | 0.0602 ± 0.0144 | 0.8573 ± 0.1961 | 30 |
| C9 (CT dose network v1, pointwise) | 30.26 ± 1.80 | 0.8456 ± 0.0718 | 0.0781 ± 0.0255 | 0.1001 ± 0.0229 | 0.8709 ± 0.1925 | 27 |
| C10 (CT dose network v1, statistics), before the variance fix | 30.17 ± 2.04 | 0.8544 ± 0.0918 | 0.0676 ± 0.0296 | 0.0934 ± 0.0171 | 0.8753 ± 0.2446 | 26 |
| C10 (CT dose network v1, statistics), variance fix | 29.89 ± 1.85 | 0.8373 ± 0.0831 | 0.0733 ± 0.0281 | 0.0997 ± 0.0176 | 0.8786 ± 0.2174 | 25 |

### Table 2

| Configuration | Network | Matching | Parameters | lambda | ms per step | M | B |
|---|---|---|---|---|---|---|---|
| C1 | VGG-19 relu3_2 | pointwise | 1,145,408 | 4.6992 | 129 | 1.0000 | 1.5092 |
| C2 | LPIPS-VGG | pointwise | 14,714,688 | 9.1625 | 141 | 0.9984 | 1.4825 |
| C3 | DISTS | statistics | 14,714,688 | 6.9836 | 147 | 0.9835 | 1.4495 |
| C4 | RadImageNet ResNet-50 | pointwise | 25,557,032 | 2.2106 | 134 | 0.9538 | 1.8118 |
| C5 | DINOv2-S | pointwise | 22,056,576 | 1.6343 | 150 | 0.9978 | 1.7518 |
| C6 | DINOv2-B | pointwise | 86,580,480 | 2.4972 | 201 | 0.9973 | 1.6694 |
| C7 | MedSigLIP-448 | pointwise | 428,565,440 | 3.7553 | 1485 | 0.9995 | 1.5346 |
| C8 | DINOv2-S | statistics | 22,056,576 | 1.9659 | 149 | 0.9973 | 1.2604 |
| C9 | CT dose network v2 (regression) | pointwise | 287,328 | 1.5493 | 126 | 0.9709 | 0.3434 |
| C10 | CT dose network v2 (regression) | statistics | 287,328 | 2.8016 | 129 | 0.8522 | 0.8633 |

- CT dose network: 287,328 parameters are the three conv blocks the loss reads. The whole network with its regression head has 1,173,857.
- lambda and ms per step are those of the final runs.
- **The Step 1.3 distance check (zero at equality, gradient, ms per batch of 8) was not rerun for v2.** The existing rows are v1 and are labelled so in `results/week1_week2_tables.md`. The seeds notebook has an optional cell for it (`CHECK_DOSE_V2_DISTANCE`).

All tables come from `python -m src.tables`. Rerunning it reproduces the files byte for byte, and every final row was checked against its per-slice CSV, run summary and training log (220 slices, 31 patients, best epoch at the lowest validation L1).

### Item by item

- [x] **Tables regenerated.** `results/table1_seed42.csv`, `table2_feature_networks.csv`, `week1_week2_tables.md`; superseded runs in `table1_seed42_superseded.csv` and a labelled section.
- [x] **Table 2 corrected** (above).
- [ ] **Reproducibility records: written, links not accessible yet.** `results/provenance.json` gives, for each of the 15 runs, the notebook, the source revision, what was uncommitted, the settings and the sha256 of `best.pth`. Limits, stated in the file:
  - C0 to C7: `d3916a2`, recorded by the notebook at run time.
  - Final C8, C9, C10: **the revision was not recorded at run time.** The code was uncommitted when they ran; the first commit that contains it is `a7108c5`, made about two hours later. The exact diff at execution time cannot be recovered.
  - `environment.json` keeps `d3916a2` with a note that it covers the Week 1 and 2 notebook only.
  - Frozen dose network v2: sha256 `69feb2a1bc7d23eb80f6bc88b57d99ea893f38284fed43876225df4b81581774`.
  - Drive: `outputs_seed42.zip` (https://drive.google.com/file/d/1gNimi2rBrtxjnrhzT77L860NNdhTXRAG/view) and `outputs_ctdosenet_followup.zip` (https://drive.google.com/file/d/13hLIkRZY1C8lywfw5n1zEBtkloPsZSmH/view). **Both are still private to my account.** The v2 dose network is only inside the second archive; there is no standalone file. The hash of the v1 Drive copy was not checked.
- [x] **Runner inconsistencies resolved; GPU check still open.**
  - `DOSE_BLOCK_SCALES` is on its own line again in the main notebook.
  - `src/run_config.py` loads the dose network through `load_dose_net`, which reads the head width from the checkpoint (6 for v1, 1 for v2).
  - `seeds_perceptual_backbone_ldct.ipynb` is the only entry point for new runs: frozen lambda values, dose network accepted only by its sha256, plain block sum, variance fix, provenance written into every `run_summary.json`. The two earlier notebooks are marked as records.
  - Checked from a fresh Python process on CPU: all three notebooks parse, and the seeds notebook runs top to bottom in `SMOKE` mode for C0, C9 and C10 on synthetic slices with batch 2 (batch 8 does not fit in 16 GB of RAM). **It has not been run on Colab or on a GPU.** The built-in reproduction check (one epoch of C9 at seed 42 against the logged epoch 1) runs first on the A100, before any seed.
- [x] **C10 validation instability investigated and documented; cause open.** `slides_and_reports/c10_validation_instability.md`.
  - Eight of 30 epochs are above twice the median validation L1 (80,138 at epoch 3, 30,478 at epoch 29), while the training loss falls smoothly.
  - The same spikes occur in C2, C3, C4, C7 and C10 v1, and never in C0 (L1).
  - The saved checkpoint reproduces the logged best validation L1 on CPU (5,717.0 against 5,716.1). At epoch 17 C10 looks like C0: no intensity offset, no failing slices.
  - No bug found in the validation path. The mechanism cannot be decided from the saved files, because only best checkpoints exist. The replay with per-slice validation diagnostics is wired (`DIAGNOSE_C10`) and **not run yet**.
- [x] **Fig 1 draft and presentation.** `figures/fig1_overview.py` writes `fig1_overview.svg`, `.pdf` and `.png` (a script in place of the `.drawio` file of the plan). `slides_and_reports/ldct_backbone_week_1_2_update.pdf` holds 10 corrected slides built from `results/`. The editable source of the original deck no longer exists, so the original PDF is kept and its result slides (6 to 12, 14, 15) are superseded by the update.
- [x] **Adoption-rule disclosure preserved.** The threshold was first 0.9 and was lowered to 0.8 after seeing the validation screen, because statistics matching (0.852) missed 0.9. The rule is not prespecified. This is now stated the same way in the follow-up notebook, `results/decision.json` (the note fields were added by hand on 2026-10-12 and are marked as such), `results/ctdosenet_followup_tables.md`, `slides_and_reports/ctdosenet_v2.md` and the update deck.

### Open before the seeds start

1. Share the two Drive archives and upload the v2 dose network as a standalone file.
2. Commit and push the working tree, so the seeds notebook records a clean revision.
3. On the A100: `REPRO_CHECK`, then `DIAGNOSE_C10` and `CHECK_DOSE_V2_DISTANCE`, then seeds 123 and 456.
