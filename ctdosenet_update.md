# CT dose network: hypothesis re-screen (CPU, 2026-10-08)

Weights: `results/seed42/ct_dose_net_v2/ct_dose_net.pt`. Screen on the 182 val slices (five doses plus matched blur), 224×224 crops, plain block sum unless stated. Scripts were run from the scratchpad; nothing was written to `results/`.
M = mean per-slice Spearman dose ordering, ratio = D(10k) / D(1000k), B = blur penalty.

## Equal-weighted blocks

Scales are 1 / mean block distance at initialisation (untrained seed-42 U-Net, 64 random train slices instead of all 899, random crops).

| Matching | Weighting | M | Ratio | B |
|---|---|---|---|---|
| pointwise | plain sum | 0.724 | 4.933 | 0.308 |
| pointwise | equal blocks | 0.726 | 4.942 | 0.310 |
| statistics | plain sum | 0.712 | 2.693 | 0.754 |
| statistics | equal blocks | 0.712 | 2.694 | 0.740 |

Scales: pointwise 1.14 / 1.04 / 1.07, statistics 1.75 / 1.93 / 2.03. Blocks were already balanced, so the weighting changes nothing. Screen only, C9 and C10 were not retrained.

## Other hypotheses (CLAUDE.md numbering)

| Matching | Variant | M | Ratio | B |
|---|---|---|---|---|
| pointwise | baseline (normalised, centre crop) | 0.724 | 4.93 | 0.308 |
| pointwise | random crops (H2) | 0.731 | 4.65 | 0.358 |
| pointwise | no channel normalisation (H1) | 0.650 | 5.81 | 0.701 |
| pointwise | no normalisation, random crops | 0.662 | 5.64 | 0.792 |
| statistics | baseline | 0.712 | 2.69 | 0.754 |
| statistics | random crops (H2) | 0.730 | 2.56 | 0.838 |
| statistics | no channel normalisation (H1) | 0.719 | 2.79 | 0.734 |
| statistics | no normalisation, random crops | 0.735 | 2.64 | 0.825 |
| statistics | old `clamp(min=1e-8)` variance (H6) | 0.712 | 2.69 | 0.754 |

Random crops: one fixed random 224×224 offset per slice (numpy seed 0), whole 512×512 slice, air included.

- **H1 (normalisation):** not supported. Removing it hurts pointwise (M 0.65, B 0.70) and barely moves statistics. Keep it.
- **H2 (crop domain gap):** not the cause. Air-containing crops score the same or slightly higher.
- **H6 (variance clamp):** no effect on the screen. Flat regions are rare in the screen crops, so the effect inside training is untested.
- **H3, H5:** not tested (H3 would move inputs away from the training distribution; H5 needs retraining). H4 and H7 are bookkeeping.

Every variant stays at M about 0.65 to 0.73, so the weak ordering is not a loss-side artefact. Remaining suspect: H5, the network itself.

## What we can do next

1. **Per-block ablation (CPU, no retrain):** screen blocks 1, 2, 3 alone and in pairs, both matchings. Shows whether one block carries or hurts the ordering.
2. **Retrain a stronger dose net** (needs a GPU or a long CPU run): validation checkpointing, augmentation, more epochs, possibly wider or deeper. Add the training loop to `src/` so it is reproducible (H7). Then re-screen. Changing the net changes C9 and C10, so rerun those rows (and lambda calibration) for every seed.
3. **Accuracy check:** compare val accuracy and the screen M for each retrain, to see whether better classification translates to a better ordering.
4. **Full-scale block scales:** recompute on all 899 train slices if equal weighting is ever adopted (the notebook v2 cell does this on GPU).
5. **Decision for the paper:** if retraining does not lift M, report C9 and C10 as the proposed network with the honest screen result and the ablation above.
