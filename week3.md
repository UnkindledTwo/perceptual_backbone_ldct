## Week 3: seeds, statistics and figures

2026-10-14. All 33 runs are done (C0 to C10 at seeds 42, 123 and 456). The statistics and Fig 2 and Fig 4 are built from them. Results are **not frozen yet** (freeze Oct 18). The work is on `main` at [`95a3842`](https://github.com/UnkindledTwo/perceptual_backbone_ldct/commit/95a3842) (on top of the Week 2 closeout `5bed4b6`).

### Summary

- Seeds 123 and 456 ran from `seeds_perceptual_backbone_ldct.ipynb` with the frozen lambda values and the frozen CT dose network v2 (sha256 `69feb2a1…81581774`). Nothing was recalibrated.
- Perceptual losses trade PSNR for perception in every configuration. C1 (VGG-19, pointwise), C2 (LPIPS-VGG, pointwise) and C8 (DINOv2-S, statistics) roughly halve LPIPS for 0.5 to 0.7 dB.
- The seed-42 lead of C8 on DISTS does not hold over three seeds. The top four are within one seed spread of each other.
- The screen as designed does not predict the trained outcome.

### Key result and takeaway

**Key result.** Over three seeds, C1 (VGG-19, pointwise), C2 (LPIPS-VGG, pointwise) and C8 (DINOv2-S, statistics) lower LPIPS from 0.104 (C0, L1) to about 0.058 at a cost of 0.5 to 0.7 dB PSNR. Every C1 to C10 loses PSNR against C0 (Family A, median 0.5 to 2.4 dB, all Holm p < 0.05).

**Takeaway.** No feature network is a clear winner on texture. Statistics matching improves grain fidelity for DINOv2-S, and medical pretraining (RadImageNet, MedSigLIP) does not help. The CT dose network matches grain statistics but stays far behind on DISTS. The training-free screen does not predict which network works: M cannot tell the networks apart, and B correlates with the trained R_H error with the opposite sign to the one the screen assumes.

### Table 1: test split (220 slices, 31 patients), 100k, seeds 42, 123, 456

| Configuration | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | R_H (→1) | \|1 − R_H\| ↓ |
|---|---|---|---|---|---|---|
| LDCT input (100k) | 23.04 | 0.7423 | 0.2843 | 0.1641 | 6.174 | 5.275 |
| C0 (L1) | **30.92 ± 0.07** | **0.8691 ± 0.0025** | 0.1043 ± 0.0010 | 0.1112 ± 0.0006 | 0.657 ± 0.026 | 0.352 ± 0.026 |
| C1 (VGG-19, pointwise) | 30.32 ± 0.22† | 0.8606 ± 0.0022 | 0.0592 ± 0.0010† | 0.0668 ± 0.0023 | 0.798 ± 0.035 | 0.226 ± 0.030† |
| C2 (LPIPS-VGG, pointwise) | 30.39 ± 0.31† | 0.8630 ± 0.0021 | 0.0563 ± 0.0045†* | **0.0645 ± 0.0024** | 0.777 ± 0.012 | 0.242 ± 0.009† |
| C3 (DISTS, statistics) | 29.81 ± 0.20† | 0.8326 ± 0.0244 | **0.0557 ± 0.0066**† | 0.0664 ± 0.0014* | 0.820 ± 0.014 | 0.222 ± 0.018† |
| C4 (RadImageNet, pointwise) | 28.70 ± 0.32† | 0.8259 ± 0.0075 | 0.0868 ± 0.0055 | 0.0847 ± 0.0009 | 0.798 ± 0.040 | 0.286 ± 0.017† |
| C5 (DINOv2-S, pointwise) | 30.10 ± 0.51† | 0.8625 ± 0.0063 | 0.0629 ± 0.0078† | 0.0707 ± 0.0068 | 0.682 ± 0.046 | 0.333 ± 0.044† |
| C6 (DINOv2-B, pointwise) | 30.20 ± 0.44† | 0.8618 ± 0.0044 | 0.0614 ± 0.0027† | 0.0713 ± 0.0028 | 0.731 ± 0.027 | 0.291 ± 0.029† |
| C7 (MedSigLIP, pointwise) | 29.84 ± 0.39† | 0.8596 ± 0.0061 | 0.0750 ± 0.0114† | 0.0737 ± 0.0070 | 0.531 ± 0.063 | 0.477 ± 0.060† |
| C8 (DINOv2-S, statistics) | 30.26 ± 0.28† | 0.8614 ± 0.0068 | 0.0579 ± 0.0112† | 0.0654 ± 0.0047 | 0.857 ± 0.017 | 0.195 ± 0.013† |
| C9 (CT dose network v2, pointwise) | 30.44 ± 0.10† | 0.8514 ± 0.0219 | 0.0673 ± 0.0031† | 0.0973 ± 0.0002 | 0.850 ± 0.018 | 0.203 ± 0.010† |
| C10 (CT dose network v2, statistics) | 29.80 ± 0.19† | 0.8475 ± 0.0039 | 0.0721 ± 0.0063† | 0.0982 ± 0.0036 | 0.951 ± 0.058 | **0.167 ± 0.026**† |

- ± is the standard deviation over the three per-seed means (not over slices, unlike the Week 2 table).
- † Holm-adjusted p < 0.05 against C0 (L1) in Family A (tested on PSNR, LPIPS and |1 − R_H| only). On |1 − R_H| the mark does not say which side is better: C7 is significantly *worse* than C0.
- \* scored with its own training metric (LPIPS for C2, DISTS for C3). DISTS is VGG-based and LPIPS is an ImageNet-CNN metric, so C1 to C3 are partly scored in their own feature space.
- Bold on R_H follows |1 − R_H|. C10 has a mean R_H of 0.951 but scatters on both sides of 1 (seed 456: 1.017), so the mean alone would overstate it.
- Per-seed values: `results/week3/table1_per_seed.csv`.

### Statistics (Step 3.2)

Paired Wilcoxon signed-rank tests on seed-averaged per-slice values (n = 220), effect size is the matched-pairs rank-biserial r. Both families were fixed before the tests and were not changed afterwards.

**Family A** (each of C1 to C10 against C0 on PSNR, LPIPS, |1 − R_H|, Holm over 30). 29 of 30 tests are significant. The exception is C4 (RadImageNet, pointwise) on LPIPS: median difference −0.0017, Holm p = 0.09, r = −0.13. Full table: `results/week3/stats_family_a.csv`.

**Family B** (Holm over 9):

| Comparison | Metric | Median difference | p (Holm) | r |
|---|---|---|---|---|
| C8 (DINOv2-S, statistics) vs C5 (DINOv2-S, pointwise) | PSNR | +0.052 | 2.8e-03 | 0.26 |
| | LPIPS | +0.0003 | 0.87 | −0.06 |
| | \|1 − R_H\| | −0.143 | 5.6e-30 | −0.90 |
| C10 (CT dose network v2, statistics) vs C9 (CT dose network v2, pointwise) | PSNR | −0.574 | 6.8e-37 | −1.00 |
| | LPIPS | +0.0013 | 0.024 | 0.21 |
| | \|1 − R_H\| | −0.060 | 8.5e-12 | −0.55 |
| C6 (DINOv2-B, pointwise) vs C5 (DINOv2-S, pointwise) | PSNR | +0.053 | 2.4e-05 | 0.36 |
| | LPIPS | +0.0005 | 0.87 | 0.03 |
| | \|1 − R_H\| | −0.034 | 1.0e-31 | −0.93 |

How to read these p-values: the tests treat 220 slices as independent, but they come from 31 patients, and seed-averaging hides the seed variance. Only one Family B effect is clearly larger than the seed spread: **statistics matching improves grain fidelity for DINOv2-S** (C8 0.195 ± 0.013 against C5 0.333 ± 0.044 on |1 − R_H|, r = −0.90). C6 against C5 on PSNR is +0.05 dB with p = 2e-5 while the seed spread is 0.5 dB, and C10 against C9 on |1 − R_H| is about 1.4 seed spreads. I would report effect sizes and seed spread next to every p-value and not lean on the small effects.

### Screen against trained outcome

Spearman over the 10 trained networks, two-sided permutation p (10,000 permutations), not corrected over the 18 rows. Outcomes are three-seed means.

| Screen score | Outcome | rho (all 10) | p | rho (without C2, C3) | p |
|---|---|---|---|---|---|
| B | \|1 − R_H\| | **+0.84** | 0.0035 | +0.71 | 0.059 |
| B | DISTS | −0.06 | 0.89 | −0.24 | 0.59 |
| B | LPIPS | +0.30 | 0.41 | +0.26 | 0.54 |
| M | \|1 − R_H\| | +0.53 | 0.12 | +0.58 | 0.14 |
| M | DISTS | −0.58 | 0.087 | −0.71 | 0.061 |
| M | LPIPS | −0.32 | 0.36 | −0.43 | 0.28 |
| Parameters | \|1 − R_H\| | +0.77 | 0.011 | +0.78 | 0.028 |

- M cannot discriminate: the pretrained networks all score about 0.99.
- B correlates **positively** with the trained R_H error, the opposite sign to the premise of the screen (a high blur penalty was supposed to mean better texture).
- My reading, not a tested result: B is probably confounded with the matching rule. The three statistics configurations (C3, C8, C10) have low B and the best R_H. On the seven pointwise networks alone rho is about 0.64 (not significant at n = 7, post hoc, not in the result files).
- Parameter count also correlates with the R_H error (rho 0.77).
- Statement for the paper: **the screen as designed does not predict the trained outcome.**

### Other observations

- **Medical pretraining does not help.** C4 (RadImageNet, pointwise) has the worst PSNR (28.70) and no LPIPS gain over C0. C7 (MedSigLIP, pointwise) is smoother than C0 (R_H 0.53 against 0.66) at about ten times the compute.
- **CT dose network: negative result on texture.** C9 and C10 stay at DISTS 0.097 to 0.098 against about 0.065 for the best group, while C10 has the best |1 − R_H|. Consistent with "matches grain statistics, blind to structure" from Week 2.
- **Seed spread grows with a perceptual term.** PSNR varies by 0.07 dB over seeds for C0 and by 0.2 to 0.5 dB for C1 to C8.
- **Validation L1 spikes** (epochs above twice the run median, seeds 42 / 123 / 456): C2 4/4/6, C3 2/5/3, C7 3/3/0, C10 8/0/1, C6 and C9 none, C0 0/0/1. The C10 instability of seed 42 does not repeat at the other seeds. The seed spread probably ties to these spikes and to checkpoint selection on a noisy validation L1; this was not tested.
- **Two unexplained SSIM dips at normal PSNR:** C3 at seed 42 (0.804 against 0.849 and 0.844) and C9 at seed 123 (0.827 against 0.858 and 0.869). Not investigated.
- The draft Family B table in the plan listed DISTS rows, but the family is defined on PSNR, LPIPS and |1 − R_H|. I kept the definition and did not add DISTS after seeing the results.

### Figures

- **Fig 2** (screen): `figures/fig2_screen.pdf`, built by `python -m figures.fig2_screen`.
- **Fig 4** (B against trained |1 − R_H|, marker size by parameter count): `figures/fig4_screen_vs_outcome.pdf`, built by `python -m figures.fig4_screen_vs_outcome`.
- **Fig 1** (overview): draft from Week 2, `figures/fig1_overview.pdf`; final version still open.
- **Fig 3** (qualitative): not built. The two slices are chosen by rule (closest to the median seed-averaged C0 PSNR, not by eye): `000039_03_01_061.png` and `000053_06_01_297.png` (`results/week3/fig3_slices.csv`). The predictions still have to be exported on Colab.

<!-- Attach fig2_screen.png and fig4_screen_vs_outcome.png here by dragging them into the comment box. -->

### Runs, provenance and environment

- **Seed 123:** `results/seed123/`, clean `094e416`. **Seed 456:** `results/seed456/`, clean `5bed4b6`. The two commits differ only in `src/tables.py`, which training does not import. Seed 42 is the set of final runs reported in Week 2 (`results/provenance.json`).
- All 22 new runs pass the integrity check: 220 test slices, 30 epochs, `best.pth` sha256 recorded and matching, no non-finite steps, same lambda values and dose-network sha256 as seed 42 (`results/week3/provenance_seeds.json`).
- GPU: one A100-SXM4 80 GB, torch 2.11.0+cu130, `cudnn.deterministic`. 3.20 GPU hours for seed 123, 3.22 for seed 456.
- Data split unchanged: patient-level 899 / 182 / 220 slices (115 / 34 / 31 patients), input `noisy_clipped_100k`, target `gt_clipped`. The test split was read once per run.
- Analysis (CPU): Python 3.12, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2 (`requirements-analysis.txt`).
- Every Week 3 output is recorded with its inputs, command, package versions and sha256 in `results/week3/manifest.json`. A snapshot check confirms that the 347 result files from Weeks 1 and 2 are unchanged.

Commands, from the repository root:

```bash
python -m src.seeds_tables                      # results/week3/: per-slice CSV, Table 1, families A and B, screen vs outcome, spikes, Fig 3 slice ids
python -m figures.fig2_screen
python -m figures.fig4_screen_vs_outcome
python -m src.record results/week3/results_before_week3.sha256   # older results unchanged
```

### Item by item

- [x] Step 3.1: seeds 123 and 456 for C0 to C10, one per-slice CSV (`results/week3/per_slice_metrics.csv`)
- [x] Step 3.2: Family A, Family B, screen-versus-outcome Spearman
- [x] Fig 2, Fig 4
- [ ] Fig 3 (export of predictions on Colab, then the figure script)
- [ ] Final Fig 1; every figure as PDF
- [ ] LaTeX export of Table 1 and Table 2
- [x] Week 3 results pushed (`95a3842`)
- [ ] Results freeze (Oct 18)

### Not done from the Week 2 proposals

- **C8 replay at seed 42** was not run. The run-time revision of the seed-42 C8 (DINOv2-S, statistics) run is still unknown. The seed 123 and 456 C8 runs are on clean commits.
- **Option B** (C2 replay with per-slice validation diagnostics) was not run, and the C10 diagnostics were not recorded for seeds 123 and 456. The spike counts above come from the training logs only.
