## Week 3 review: what was done before the freeze

2026-10-15. Thanks for the review. Below, each item of your list gets a short summary of what was done, where the output is, and what is still open. All new outputs are in `results/week3/`, and every number below is read from the generated files. The commands that rebuild everything are at the end.

Code: [`a9aaa30`](https://github.com/UnkindledTwo/perceptual_backbone_ldct/commit/a9aaa30). All outputs were rebuilt on that clean commit; `results/week3/manifest.json` records it for every step.

### Item by item

- [x] **Patient-level statistics.** Families A and B now also run on patient means: seeds averaged per slice, then slices averaged per patient (n = 31), with the same paired Wilcoxon test and the same Holm families (30 and 9 tests). The planned slice-level families are unchanged, and the script asserts that they reproduce the earlier tables exactly. The new table puts effect sizes and the per-seed mean differences with their seed std next to every p-value (`stats_families_slice_and_patient.csv`, section in `week3_tables.md`).
  - **Family A:** 30 of 30 tests are significant at patient level, against 29 of 30 at slice level. The one change is C4 (RadImageNet, pointwise) on LPIPS: Holm p = 0.021, r = −0.47 at patient level, against p = 0.09 at slice level. The headline holds: every C1 to C10 loses PSNR against C0.
  - **Family B:** 3 of 9 tests are significant at patient level, against 7 of 9 at slice level. As you expected, the small effects drop out:
    - C8 (DINOv2-S, statistics) vs C5 (DINOv2-S, pointwise) on PSNR: p = 0.91
    - C6 (DINOv2-B, pointwise) vs C5 on PSNR: p = 0.81
    - C10 (CT dose network v2, statistics) vs C9 (CT dose network v2, pointwise) on LPIPS: p = 0.91
    - C10 vs C9 on |1 − R_H|: p = 0.075
  - **What survives in Family B:**
    - **C8 vs C5 on |1 − R_H|:** p = 4.9e-4, r = −0.77. C8 is better at every seed: −0.171, −0.099 and −0.147.
    - **C10 vs C9 on PSNR:** −0.54 dB, p = 8.4e-9.
    - **C6 vs C5 on |1 − R_H|:** p = 4.0e-4, r = −0.78. This one changes sign at seed 123 (+0.017), and its seed std of 0.058 is larger than the mean difference of 0.042, so I would not claim it.

  The robust positive result is therefore still C8 against C5 on texture.

- [x] **Direction of the texture mark.** In Table 1, † now means significantly better than C0 (L1) and ‡ significantly worse. C7 (MedSigLIP, pointwise) on |1 − R_H| now carries ‡, and every PSNR row carries ‡. A second table gives signed differences to C0 per configuration (mean ± seed std of the per-seed differences: `table1_deltas_vs_c0.csv`).

- [x] **Screen against outcome.** The section now states that the 10 points are not 10 independent networks:
  - C5 and C8 share DINOv2-S.
  - C9 and C10 share the CT dose network v2.
  - C1 to C3 are all VGG-based.

  The pointwise-only correlation is now computed by the script, not by hand, and labelled "post hoc, exploratory" (`screen_vs_outcome_exploratory.csv`). B against |1 − R_H| gives ρ = 0.64, p = 0.14 (n = 7). Parameter count against |1 − R_H| gives ρ = 0.89, p = 0.013, uncorrected and also exploratory. The paper statement stays the same: the screen as designed does not predict the trained outcome.

- [x] **Disclosures** (`results/week3/disclosures.md`, generated from `decision.json` and the provenance files):
  1. The v2 adoption threshold was lowered from 0.9 to 0.8 after seeing the screen, quoted from `decision.json`.
  2. An ablation table puts the pre-fix C8, v1 C9, v1 C10 and C10 v1 with the variance fix next to the final seed-42 run of each (`table_ablation_superseded.csv` / `.tex`).
  3. **The source revision of the final seed-42 C8 run is closed by disclosure, without a replay.** The revision was not recorded at run time; the first commit with that code is `a7108c5`. The C8 runs at seeds 123 and 456 ran on clean recorded commits (`094e416`, `5bed4b6`), and the three seeds are shown side by side. It goes into the paper as a limitation: one of the 33 runs has no recorded revision.
  4. The eval-mode validation instability, with the spike counts per seed.

- [x] **Fig 3** (`figures/fig3_qualitative.pdf`). The figure shows one slice at all three seeds in one grid:
  - Rows: seeds 42, 123 and 456, plus one row of |prediction − NDCT| at seed 42.
  - Columns: LDCT, C0, C1, C3, C5, C7, **C8** (next to C5), C10 and NDCT.
  - The slice is the first one in `fig3_slices.csv`, chosen by the median rule.
  - All images use the same grey range [0, 1], and all error maps the same scale (0 to 0.1).

  The predictions were exported on Colab from the clean commit `a9aaa30` (`results/week3/fig3/`). The export refuses any checkpoint whose sha256 differs from the record and checks every prediction against the logged PSNR. All 21 checkpoints match. `python -m figures.fig3_qualitative` checks the file hashes and the PSNR again before drawing.

  At full-slice size, the grain differences between the perceptual losses are hard to see in print. The error row shows them more clearly than the images. A zoomed region would help if you want the texture to be visible; it is not done, because the region would have to be chosen by a rule, not by eye.

- [x] **Fig 1 final, Table 2 final, LaTeX export.**
  - **Fig 1:** updated to the final protocol: CT dose network v2 with 0.29 M feature-extractor parameters, three seeds, and the per-slice and per-patient tests. It is now built reproducibly and recorded in the manifest.
  - **Table 2** (`table2_final.csv` / `.tex`) lists for each network:
    - 287,328 feature-extractor parameters for the CT dose network, with the 1,173,857 including the head in a footnote
    - the final λ
    - the measured distance time per batch of 8
    - the training step time and GPU hours per run, averaged over the three seeds
    - M and B
  - **LaTeX:** `table1_final.tex`, `table2_final.tex` and `table_ablation_superseded.tex` use booktabs. They have not been compiled yet, because there is no LaTeX installation on the analysis machine.

- [x] **Fig 2, Fig 3 and Fig 4** are attached below.

<!-- Attach figures/fig2_screen.png, figures/fig3_qualitative.png and figures/fig4_screen_vs_outcome.png here. -->

### Reproduce

From the repository root, CPU only, in the analysis environment (`requirements-analysis.txt`: Python 3.12, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2):

```bash
python -m src.seeds_tables          # Table 1 (marks † / ‡), deltas, families A and B per slice and per patient,
                                    # screen vs outcome (+ exploratory), Table 2, ablation, disclosures, LaTeX
python -m figures.fig1_overview
python -m figures.fig2_screen
python -m figures.fig4_screen_vs_outcome
python -m figures.fig3_qualitative  # after the Colab export below
python -m src.record results/week3/results_before_week3.sha256   # the 347 Week 1 and 2 result files are unchanged
```

Fig 3 export (Colab, A100, no training): in `seeds_perceptual_backbone_ldct.ipynb`, set `EXPORT_FIG3 = True`, `REPRO_CHECK = False` and `TRAIN_CONFIGS = []`. Put the 21 checkpoints under `FIG3_CHECKPOINTS` as the cell's markdown describes, run the setup cells and the export cell, and copy `outputs/perceptual_backbone_seeds/fig3/` to `results/week3/fig3/`.

**Checks that ran:**
- Re-running `src.seeds_tables` gives byte-identical results: every existing Week 3 CSV and `provenance_seeds.json` comes out unchanged.
- A second run of all commands reproduces every new file byte for byte. Figures are written without timestamps.
- `results/week3/manifest.json` records the command, inputs, outputs, sha256, package versions and source state of every step.
- The snapshot check reports the 347 older result files unchanged.
