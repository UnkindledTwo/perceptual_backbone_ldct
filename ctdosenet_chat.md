# CT dose network (C9, C10): review, tests and next steps

Notes collected from the code review and the diagnostic tests. Test runs were CPU-only, on subsets of the validation split, with the saved seed-42 weights. No retraining was done, and no project files were changed by the tests.

## 1. Observed results (seed 42)

- Screen: ordering M = 0.724 (pointwise) / 0.712 (statistics); blur penalty B = 0.308 / 0.754.
- Dose classification, validation: 6 classes 78.5%; clean vs 1000k 90.5%. Main confusions: 500k vs 1000k (286 patches), 50k vs 100k (132).
- Test split: C9 PSNR 30.26, LPIPS 0.078, DISTS 0.100, R_H 0.871. C10 PSNR 30.17, LPIPS 0.068, DISTS 0.093, R_H 0.875.
- Best texture match (R_H) of all runs, but the worst DISTS among the perceptual losses.

## 2. Code review (no crash-level bug found)

Correct: label order is consistent between training and the screen; BatchNorm is in eval mode when used as a loss; gradients reach the prediction; train and val patients are disjoint; the screen harness reproduces the reported numbers exactly.

Implementation notes:
- `DoseNetFeatures` ignores `prepare()`; `input_size` and the mean/std buffers are unused. Inputs are raw 1-channel crops. This matches training but differs from the plan text ("resized to the network's input size").
- Reported 1,175,142 parameters include the classifier head. Feature blocks: 287,328 (24.5%). Head: 887,814.
- `local_stats` clamps variance at 1e-8 instead of 0, which zeroes the gradient in flat windows. It applies to every network using statistics matching (C8 and C10).
- Training recipe: 10 epochs, no validation checkpointing (last epoch saved), no augmentation. Final train accuracy about 87%, validation about 79%.
- The training loop exists only in the notebook (cells 28 to 30); `ct_dose_net.pt` is gitignored.

## 3. Hypotheses tested

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Channel L2 normalisation removes dose information | Not the main cause | Unnormalised squared difference: M 0.67, B 0.62. Magnitude-only: M 0.74. Neither fixes the ordering. |
| Crop domain gap (random 224 crops include air) | Contradicted | Crops with air: M = 1.00. Body-only crops (what it trained on): M = 0.31. About 51% of random loss crops are body-only. |
| Unused resize (128 patch vs 224 crop) | No effect | Head accuracy 79.2% on 224 crops vs 77.3% on 128 patches. |
| Parameter count misleading | Confirmed | See section 2. |
| Shallow features / thin recipe | Partly supported | See section 4. |
| Variance clamp removes gradient | Confirmed (C10 only) | Share of windows at the clamp on body-only clean crops: 25% (block 1), 43% (block 2), 52% (block 3). |
| BatchNorm eval vs train mismatch | Minor | Accuracy 78.9% (running stats) vs 82.1% (batch stats of 64). Batch and running means differ by at most 0.05 std. |

## 4. Main finding: the failure sits in blocks 2 and 3

- Block 1 alone orders doses almost perfectly (M = 0.999; raw distance ratio 35.7 against 4.9 for the summed distance).
- Blocks 2 and 3 give M 0.65 to 0.68 on the full screen and about 0.17 on body-only crops. Mean distances for 10k, 50k and 100k are close (block 2: 0.88, 0.77, 0.73), so per-crop order is scrambled. The features saturate: they separate "noisy" from "clean" but do not grade noise level.
- The summed distance is dominated by blocks 2 and 3.
- Per-image batch statistics give M = 0.997 but collapse classification to chance (16.7%), so that is not a usable fix.

Dose-net distance to the target, body-only validation crops:

| Output | Block 1 | Block 2 | Block 3 | Sum |
|---|---|---|---|---|
| input (100k) | 0.670 | 0.997 | 1.114 | 2.781 |
| C0 (L1) | 0.173 | 0.218 | 0.406 | 0.797 |
| C5 (DINOv2-S, pointwise) | 0.169 | 0.204 | 0.372 | 0.745 |
| C8 (DINOv2-S, statistics) | 0.159 | 0.120 | 0.193 | 0.472 |
| C9 (CT dose network, pointwise) | 0.159 | 0.071 | 0.089 | 0.320 |
| C10 (CT dose network, statistics) | 0.174 | 0.083 | 0.115 | 0.372 |

C9 lowers the summed distance by 2.5x relative to L1, almost entirely in blocks 2 and 3. Block 1 barely changes. Yet DISTS only moves from 0.112 to 0.100, while the other perceptual losses reach 0.06 to 0.07. The loss can be minimised without improving structure.

Dose-net head labels (128x128 centre sub-patch, counts of 93 crops): L1 outputs read as "clean" in 92 cases; real normal-dose images in 80; C8 outputs read noisier (57 clean, 24 as 500k, 9 as 100k). The head cannot tell smoothed from truly clean.

## 5. Conclusion

The weak result comes mainly from the features, not from a bug. A classifier trained on 6 classes saturates in its deeper blocks and does not encode anatomy or structure, so matching its features does not enforce structure. Block 1 (grain) is a good ruler; blocks 2 and 3 are not, but they dominate the sum.

## 6. Possible improvements (ranked)

1. Change which blocks enter the distance: block 1 only, or equal per-block weighting. Cost: one U-Net run (about 0.15 GPU hours). Caveat: block 1 has B about 0.37.
2. Train graded features: ordinal or log-dose regression target, or a pairwise ranking loss, instead of 6-way cross-entropy.
3. Add structure information: multi-task or self-supervised head, more capacity or epochs with validation checkpointing.
4. Fix the statistics variant: use a smooth floor such as `sqrt(var + eps)` instead of `clamp(min=1e-8)`.
5. If nothing closes the gap, report it as a finding: a task-trained CT network matches grain but not structure.

## 7. Impact on the paper plan

- Low risk: diagnosis only; block weighting or block-1-only (amend the Protocol table, Step 2.2 and Table 2; recalibrate lambda; rerun C9, C10 and their screen rows).
- Needs an explicit amendment: ordinal or ranking training (changes the pretraining description and the Step 2.2 checks); the variance-floor fix must apply to C8 and C10 together, with lambda recalibrated and both rerun, to keep the matched pairs consistent.
- Would break the plan: a hybrid objective (one d_phi per configuration), per-network lambda tuning, choosing a variant on the test split, adding configurations without updating the Holm counts and families, training on blurred patches.
- Handling: decide early in Week 3 before seeds 123 and 456, choose on validation data and the screen, keep the original C9 and C10 as a "v1" ablation, disclose every variant tried, and update the plan and compute budget. The plan calls the screen a prediction, not a filter, so describe any redesign as a response to the diagnosed saturation.
- Note: `paper_plan.md` appeared after this analysis; this section was based on `paper_plan(1).md`.

## 8. Test method (for reproduction)

- Data: validation split of `dataset_biailab_I2I_2025v1`; weights from `results/seed42/`.
- Screen variants: all 182 validation slices, centre crop 144:368, as in `src/screen.py`.
- Crop groups: random 224 crops from every third slice, split by body fraction (at least 0.9, 0.5 to 0.9, below 0.5), body mask by threshold 0.05.
- Output analysis: U-Net outputs of C0, C5, C8, C9, C10 on every sixth slice, three body-only crops each.
- Classification checks: 2,184 validation patches of 128x128 (2 per slice, 6 sources each), body fraction at least 0.9.
- Limits: CPU only, subsets of the validation split, no retraining, so the proposed fixes are untested.
