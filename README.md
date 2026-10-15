# Which Feature Network Makes a Good Perceptual Loss for LDCT?

Benchmark code for the ISBI 2027 paper. A fixed U-Net is trained for low-dose CT denoising with
`L1 + lambda * d_phi(crop(pred), crop(target))`; only the feature network `phi` and the matching rule
(pointwise or local statistics) change between the 11 configurations C0 to C10. Protocol and plan: `paper_plan.md`.

## Results (test split, 220 slices, 100k, seeds 42, 123, 456)

| Configuration | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | R_H (→1) | \|1 − R_H\| ↓ |
|---|---|---|---|---|---|---|
| LDCT input (100k) | 23.04 | 0.7423 | 0.2843 | 0.1641 | 6.174 | 5.275 |
| C0 (L1) | **30.92 ± 0.07** | **0.8691 ± 0.0025** | 0.1043 ± 0.0010 | 0.1112 ± 0.0006 | 0.657 ± 0.026 | 0.352 ± 0.026 |
| C1 (VGG-19, pointwise) | 30.32 ± 0.22‡ | 0.8606 ± 0.0022 | 0.0592 ± 0.0010† | 0.0668 ± 0.0023 | 0.798 ± 0.035 | 0.226 ± 0.030† |
| C2 (LPIPS-VGG, pointwise) | 30.39 ± 0.31‡ | 0.8630 ± 0.0021 | 0.0563 ± 0.0045†* | **0.0645 ± 0.0024** | 0.777 ± 0.012 | 0.242 ± 0.009† |
| C3 (DISTS, statistics) | 29.81 ± 0.20‡ | 0.8326 ± 0.0244 | **0.0557 ± 0.0066**† | 0.0664 ± 0.0014* | 0.820 ± 0.014 | 0.222 ± 0.018† |
| C4 (RadImageNet, pointwise) | 28.70 ± 0.32‡ | 0.8259 ± 0.0075 | 0.0868 ± 0.0055 | 0.0847 ± 0.0009 | 0.798 ± 0.040 | 0.286 ± 0.017† |
| C5 (DINOv2-S, pointwise) | 30.10 ± 0.51‡ | 0.8625 ± 0.0063 | 0.0629 ± 0.0078† | 0.0707 ± 0.0068 | 0.682 ± 0.046 | 0.333 ± 0.044† |
| C6 (DINOv2-B, pointwise) | 30.20 ± 0.44‡ | 0.8618 ± 0.0044 | 0.0614 ± 0.0027† | 0.0713 ± 0.0028 | 0.731 ± 0.027 | 0.291 ± 0.029† |
| C7 (MedSigLIP, pointwise) | 29.84 ± 0.39‡ | 0.8596 ± 0.0061 | 0.0750 ± 0.0114† | 0.0737 ± 0.0070 | 0.531 ± 0.063 | 0.477 ± 0.060‡ |
| C8 (DINOv2-S, statistics) | 30.26 ± 0.28‡ | 0.8614 ± 0.0068 | 0.0579 ± 0.0112† | 0.0654 ± 0.0047 | 0.857 ± 0.017 | 0.195 ± 0.013† |
| C9 (CT dose network v2, pointwise) | 30.44 ± 0.10‡ | 0.8514 ± 0.0219 | 0.0673 ± 0.0031† | 0.0973 ± 0.0002 | 0.850 ± 0.018 | 0.203 ± 0.010† |
| C10 (CT dose network v2, statistics) | 29.80 ± 0.19‡ | 0.8475 ± 0.0039 | 0.0721 ± 0.0063† | 0.0982 ± 0.0036 | 0.951 ± 0.058 | **0.167 ± 0.026**† |

Mean ± standard deviation over the three per-seed means. Bold: best trained configuration. † significantly better and
‡ significantly worse than C0 (L1) (Holm-adjusted p < 0.05, paired Wilcoxon test on seed-averaged per-slice values;
tested on PSNR, LPIPS and |1 − R_H|). * scored with its own training metric. All tests (per slice as planned, per patient
as a sensitivity check), signed differences to C0, the screen-versus-outcome correlations, Table 2, the ablation of
superseded runs and the validation-spike counts are in `results/week3/week3_tables.md`; disclosures in
`results/week3/disclosures.md`; LaTeX tables in `results/week3/*.tex`. The results are not frozen yet (freeze: 2026-10-18).

## Rebuilding the Week 3 tables and figures (CPU, no data)

```
python -m venv ~/.venvs/ldct-analysis && ~/.venvs/ldct-analysis/bin/pip install -r requirements-analysis.txt
python -m src.seeds_tables            # results/week3/*.csv, *.tex, week3_tables.md, disclosures.md, provenance_seeds.json
python -m figures.fig1_overview       # figures/fig1_overview.{pdf,svg,png}
python -m figures.fig2_screen         # figures/fig2_screen.{pdf,svg,png}
python -m figures.fig4_screen_vs_outcome
python -m figures.fig3_qualitative    # needs results/week3/fig3/ from the EXPORT_FIG3 cell of the seeds notebook (Colab)
python -m src.record results/week3/results_before_week3.sha256   # older result files unchanged?
```

Every command writes its inputs, outputs and their sha256, the package versions and the source revision to
`results/week3/manifest.json`. Outputs are byte-identical on a rerun. Nothing outside `results/week3/` and
`figures/` is written; `python -m src.tables` (Weeks 1 and 2, seed 42) rewrites its own older files and is not part
of this step.

## Layout

- `seeds_perceptual_backbone_ldct.ipynb`: the only entry point for training runs (seeds 123 and 456, A100).
- `src/`: modules (`seeds_tables.py`, `stats.py`, `record.py`, `latex.py` for Week 3; `tables.py` for the seed-42 tables).
- `results/`: per-run metrics, logs and summaries (`seed42/`, `seed123/`, `seed456/`), `week3/` for the three-seed analysis,
  `provenance.json` and `week3/provenance_seeds.json` for the source revision and checkpoint hash of every run.
- `figures/`: figure scripts and their outputs.
- Model weights (`*.pt`, `*.pth`) are not in the repository.
