## Week 2 closeout: reply

2026-10-12. Everything below is seed 42. The work is on `main` at [`5bed4b6`](https://github.com/UnkindledTwo/perceptual_backbone_ldct/commit/5bed4b6) (closeout in `094e416`, GPU verification on top).

### Summary

- All closeout items are done. Two things remain open and are listed at the end: the source revision of the final C8 run cannot be shown the way it now can for C9 and C10, and the full 11-configuration loop of the new runner has not run yet.
- The committed code reproduces the final C9 and C10 runs exactly on the A100, so the runner for seeds 123 and 456 is verified against seed 42.
- The C10 validation spikes are not a bug. They are a BatchNorm train/eval mismatch on the bright validation slices. The saved checkpoint is unaffected.

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

| Configuration | Network | Matching | Parameters | lambda | ms per step | ms per batch of 8 | M | B |
|---|---|---|---|---|---|---|---|---|
| C1 | VGG-19 relu3_2 | pointwise | 1,145,408 | 4.6992 | 129 | 11.1 | 1.0000 | 1.5092 |
| C2 | LPIPS-VGG | pointwise | 14,714,688 | 9.1625 | 141 | 27.1 | 0.9984 | 1.4825 |
| C3 | DISTS | statistics | 14,714,688 | 6.9836 | 147 | 26.4 | 0.9835 | 1.4495 |
| C4 | RadImageNet ResNet-50 | pointwise | 25,557,032 | 2.2106 | 134 | 20.6 | 0.9538 | 1.8118 |
| C5 | DINOv2-S | pointwise | 22,056,576 | 1.6343 | 150 | 31.5 | 0.9978 | 1.7518 |
| C6 | DINOv2-B | pointwise | 86,580,480 | 2.4972 | 201 | 79.4 | 0.9973 | 1.6694 |
| C7 | MedSigLIP-448 | pointwise | 428,565,440 | 3.7553 | 1485 | 1370.5 | 0.9995 | 1.5346 |
| C8 | DINOv2-S | statistics | 22,056,576 | 1.9659 | 149 | 28.9 | 0.9973 | 1.2604 |
| C9 | CT dose network v2 (regression) | pointwise | 287,328 | 1.5493 | 126 | 8.7 | 0.9709 | 0.3434 |
| C10 | CT dose network v2 (regression) | statistics | 287,328 | 2.8016 | 129 | 10.9 | 0.8522 | 0.8633 |

- CT dose network: 287,328 parameters are the three conv blocks the loss reads. The whole network with its regression head has 1,173,857.
- lambda and ms per step are those of the final runs.
- The Step 1.3 distance check was rerun for v2 on 2026-10-12 (`results/zoo_checks_v2.csv`): distance 0 at equality and the gradient reaches the prediction for both matchings. The v1 rows are kept and labelled as the ablation.

All tables come from `python -m src.tables`. Every final row was checked against its per-slice CSV, run summary and training log (220 slices, 31 patients, best epoch at the lowest validation L1).

### Item by item

- [x] **Tables regenerated.** `results/table1_seed42.csv`, `table2_feature_networks.csv`, `week1_week2_tables.md`; superseded runs in `table1_seed42_superseded.csv` and a labelled section.
- [x] **Table 2 corrected** (above).
- [x] **Reproducibility records.** `results/provenance.json` gives, for each of the 15 runs, the notebook, the source revision, what was uncommitted, the settings and the sha256 of `best.pth`.
  - C0 to C7: `d3916a2`, recorded by the notebook at run time.
  - Final C8, C9, C10: **the revision was not recorded at run time.** The code was uncommitted when they ran; the first commit that contains it is `a7108c5`, made about two hours later.
  - C9 and C10 are now tied to a commit after the fact: `094e416` with no uncommitted source files reproduces them exactly (next item). C8 and C0 to C7 were not replayed.
  - `environment.json` keeps `d3916a2`, with a note that it covers the Week 1 and 2 notebook only.
  - Drive, anyone with the link can view: [`outputs_seed42.zip`](https://drive.google.com/file/d/1gNimi2rBrtxjnrhzT77L860NNdhTXRAG/view) and [`outputs_ctdosenet_followup.zip`](https://drive.google.com/file/d/13hLIkRZY1C8lywfw5n1zEBtkloPsZSmH/view). They hold every `best.pth`.
  - Frozen dose network v2: sha256 `69feb2a1bc7d23eb80f6bc88b57d99ea893f38284fed43876225df4b81581774`, inside the second archive. A standalone copy exists on Drive but is still private. The hash of the v1 Drive copy was not checked.
- [x] **Runner inconsistencies resolved and checked from a fresh kernel.**
  - `DOSE_BLOCK_SCALES` is on its own line again in the main notebook.
  - `src/run_config.py` loads the dose network through `load_dose_net`, which reads the head width from the checkpoint (6 for v1, 1 for v2).
  - `seeds_perceptual_backbone_ldct.ipynb` is the only entry point for new runs: frozen lambda values, dose network accepted only by its sha256, plain block sum, variance fix, commit and source hashes written into every `run_summary.json`. The two earlier notebooks are marked as records.
  - Fresh Colab kernel on the A100, clean clone of `094e416`: one epoch of C9 (CT dose network v2, pointwise) at seed 42 gives training loss 2.792584854782092 and validation L1 9280.567372081045, a relative difference of 0.0 to the logged epoch 1 (`results/repro_seed42/repro_check.csv`).
  - Since `094e416` the only change in `src/` is a label in `src/tables.py`, which training does not import.
- [x] **C10 validation instability investigated: not a bug.** `slides_and_reports/c10_validation_instability.md`, data in `results/seed42/diag_C10_v2/`. Validation split only; the test split was not read.
  - Eight of 30 epochs are above twice the median validation L1 of 8,579 (80,138 at epoch 3, 30,478 at epoch 29), while the training loss falls smoothly.
  - A full replay with per-slice diagnostics gives the committed validation L1 at all 30 epochs and a byte-identical `best.pth`. The spikes are deterministic and reproducible.
  - At a spike, 40 to 50 of the 182 validation slices are predicted far too bright in eval mode (mean prediction 0.81 against a mean target of 0.43 at epoch 29). They are the same bright slices each time: 43 fail in at least six of the eight spikes, from 17 of 34 patients. They carry 62 to 87% of the total L1. The other slices are unaffected. Epoch 3 is wider, with 115 failing slices.
  - With BatchNorm on batch statistics the same weights show no spike: that L1 falls steadily from 17,611 to 9,030. It is a train/eval BatchNorm mismatch on bright slices, and the training loss, computed in batch-statistics mode, does not see it.
  - Epoch 17, the saved checkpoint, has no failing slice. The C10 row is valid as reported.
  - The same kind of spike appears in the logs of C2, C3, C4, C7 and C10 v1, and never in C0 (L1). Those runs were not replayed, so it is not confirmed that they fail on the same slices.
- [x] **Fig 1 draft and presentation.** `figures/fig1_overview.py` writes `fig1_overview.svg`, `.pdf` and `.png` (a script in place of the `.drawio` file of the plan); it describes the final regression-v2 protocol. `slides_and_reports/ldct_backbone_week_1_2_update.pdf` holds 10 corrected slides built from `results/`. The editable source of the original deck no longer exists, so the original PDF is kept and its result slides (6 to 12, 14, 15) are superseded by the update.
- [x] **Adoption-rule disclosure preserved.** The threshold was first 0.9 and was lowered to 0.8 after seeing the validation screen, because statistics matching (0.852) missed 0.9. The rule is not prespecified. This is stated the same way in the follow-up notebook, `results/decision.json` (the note fields were added by hand on 2026-10-12 and are marked as such), `results/ctdosenet_followup_tables.md`, `slides_and_reports/ctdosenet_v2.md` and the update deck.

### What is still open

1. **C8 revision.** C8 (DINOv2-S, statistics) is the headline configuration and its run-time revision is unknown. It was not replayed.
2. **Full runner loop.** The seeds notebook has run its checks and two single configurations on the GPU, not yet all 11 in one session. Seed 123 is its first full run.
3. **Small items carried over:** `RADIMAGENET_STATS` is still `None` (ImageNet statistics; to confirm against the weights source), the "GPU hours available" cell is blank, the Step 1.2 audit table of #208 and #261 is not in the outputs, `README.md` is empty.

### What we can do next

Proposed, in this order. Steps 1 and 3 need a small addition to the seeds notebook; neither changes how a run is trained.

1. **Replay C8 at seed 42 from the committed code before the seeds** (about 11 minutes on the A100). If it matches the log as C9 and C10 did, all three follow-up runs are tied to a commit and the provenance gap is closed for the headline row. If it does not match, we learn that before two more seeds are built on it.
2. **Run seeds 123 and 456 with the frozen configuration**, as planned (about 3.2 GPU hours per seed). No recalibration, no change to the dose network.
3. **Record the per-slice validation diagnostics for C10 in both seeds.** They run after each epoch's validation and do not touch the model, the checkpoint rule or the test split. The seed-42 replay had them switched on and still reproduced the log exactly, so the rows stay comparable. This shows whether the instability and the early best epoch repeat.
4. **Week 3 analysis as planned:** Wilcoxon tests with Holm correction (families A and B), Spearman tests of screen against outcome, Fig 2 to Fig 4, freeze on Oct 18.

On the C10 instability itself there are three options. I propose A.

| Option | Cost | Effect on the benchmark |
|---|---|---|
| **A. Keep the Protocol and disclose.** Report that the eval-mode validation L1 of C10 was unstable, name the cause, and state that the checkpoint rule selected epoch 17. | none | None. Every row stays comparable. |
| B. In addition to A, replay C2 (LPIPS-VGG, pointwise) at seed 42 with the same diagnostics. | about 11 minutes | None. It tells us whether the spikes of a pretrained network hit the same bright slices. If so, the paper can describe the instability as a property of the U-Net, not of the dose network. |
| C. Remove the cause: another normalisation or BatchNorm momentum in the U-Net, or BatchNorm statistics recomputed before validation. | all 33 runs again, about 11 GPU hours, plus recalibration | Changes the Protocol for every row, including what the learning-rate scheduler sees. It does not fit the Oct 18 freeze unless it is decided now. |

Two points I would like your view on:

- **Option B.** It is cheap and I would run it together with step 1 unless you see a reason not to.
- **The screen.** The CPU diagnosis showed that the dose-network distance is nearly blind to constant offsets and low-frequency error, which M and B do not measure. A structure term in the screen would be a post hoc addition. I would keep the prespecified M and B for the Spearman tests and report a structure term only as a labelled exploratory analysis. Is that the scope you want?
