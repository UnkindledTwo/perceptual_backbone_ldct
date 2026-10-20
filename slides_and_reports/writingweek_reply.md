## Writing week: Fig 3 done, two questions on the plan

2026-10-19. Thanks for updating the plan. I will write from [`writing_plan.md`](https://github.com/itu-biai/dlct-docs/blob/main/projects/perceptual_backbone_ldct/writing_plan.md) at `2d79f95`, not from the copy in the issue body.

Code: [`dfb75c5`](https://github.com/UnkindledTwo/perceptual_backbone_ldct/commit/dfb75c5); `results/week3/manifest.json` records the Fig 3 step on that clean commit.

### Fig 3

- [x] **One seed, zoom row, error row** (`figures/fig3_qualitative.pdf`, attached below).
  - Rows: full slice, 96 × 96 px zoom, |prediction − NDCT|. Columns as before: LDCT, C0, C1, C3, C5, C7, C8, C10, NDCT.
  - Seed 42 only. The seed 123 and 456 arrays stay in `results/week3/fig3/` but are no longer drawn.
  - Same slice (first row of `fig3_slices.csv`, median C0 PSNR), same grey range [0, 1] for images and zooms, same error scale (0 to 0.1).
  - **Zoom rule** (computed on the NDCT, `zoom_window` in `figures/fig3_qualitative.py`):
    1. Body mask: NDCT > 0.05, morphological opening, holes filled, largest connected component.
    2. Soft tissue: body pixels with NDCT between 0.35 and 0.75 (the soft-tissue peak of the grey-value histogram, between fat and contrast or bone).
    3. Among the 96 px windows on an 8 px grid that are at least 90% soft tissue, take the one with the median NDCT standard deviation; ties go to the first window in row-major order.

    For this slice the rule selects a liver region (rows 168 to 263, columns 120 to 215) with a few portal vessels. The box is drawn on every top-row panel, the rule is printed on the figure, and the parameters and the selected corner are in the manifest.
  - Caption sentence: *"Zoom: the 96 × 96 px window with the median NDCT standard deviation among windows that are at least 90 % soft tissue (NDCT 0.35 to 0.75 inside the body mask)."*

  What the zoom shows, on this one slice (observations, not tested):
  - C0 (L1) is visibly smooth; every perceptual loss keeps grain.
  - C10 (CT dose network v2, statistics) has the coarsest grain.
  - C1 (VGG-19, pointwise) shows a faint regular checkerboard pattern.
  - C5 (DINOv2-S, pointwise) and C8 (DINOv2-S, statistics) look alike at this size.

  The rebuilt figure keeps the same checks as before: export hashes and PSNR against the logged metrics. A rerun gives identical bytes.

### Other items before Monday

- [x] **Novelty re-check.** Search strings 1 to 7 of `novelty.md` were re-run on arXiv on Oct 19. No paper submitted since Sep 28 matches; the verdict and Introduction P4 stay as they are.
- [x] **Reference metadata:**
  - ALDEN is MICCAI 2025 (doi 10.1007/978-3-032-04937-7_2).
  - LEDA is IEEE BIBM 2024 (doi 10.1109/BIBM62325.2024.10822519).
  - RadImageNet is doi 10.1148/ryai.210315.
  - The Lab-Rasool RadImageNet model card on Hugging Face asks for standard ImageNet normalization, which is what the code used, so C4 (RadImageNet, pointwise) was trained with the right input statistics.
  - Still open: whether to cite the withdrawn Fang et al.

### Questions on the plan

Following your rule (no number in `results/week3/`, so mark it for discussion), two sentences in `writing_plan.md` say a little more than the results show:

1. **Implementation:** "eval-mode validation L1 spikes (checkpoint selection unaffected)". This was not tested. We know that the best epochs have no failing slices for C10 (CT dose network v2, statistics) at seed 42 (GPU replay), but not that the selected checkpoint would be the same without the spikes. The seed spread may partly come from checkpoint selection on a noisy validation L1. Should the sentence say only that the spikes occur, with the counts per seed?
2. **Fig 3 and Fig 4 paragraph:** "in the zoom row, DINOv2-S with statistics keeps grain closer to NDCT than DINOv2-S pointwise". In the final zoom, C5 and C8 look alike, and it is one slice. Should the paragraph give the Family B result (|1 − R_H| 0.333 to 0.195, patient-level p = 4.9e-4, r = −0.77) as the evidence and use the zoom only to show that L1 removes the grain?

### Reproduce

```bash
python -m figures.fig3_qualitative   # needs results/week3/fig3/seed42/ (already exported on Oct 15 from a9aaa30)
```

<!-- Attach figures/fig3_qualitative.png here. -->
