# Validation instability of C10 (CT dose network v2, statistics), seed 42

Written 2026-10-12, updated the same day with the GPU replay. Only training and validation data are used here; the test split is not read.

## Verdict

**Not a bug, and the run is exactly reproducible.** The GPU replay of 2026-10-12 (commit `094e416`, A100) gives the committed validation L1 at all 30 epochs with a relative difference of 0.0, and its `best.pth` has the same sha256 as the original. The spikes are an eval-mode failure on the bright validation slices: at a spike epoch about 40 to 50 of the 182 slices are predicted far too bright when BatchNorm uses its running statistics, while the same weights give a normal error when BatchNorm uses batch statistics. The other slices are unaffected. Epoch 17, the saved checkpoint, has no failing slice.

## What the log shows

`val_l1` is the absolute error summed over the 512×512 pixels of a slice and averaged over the 182 validation slices, so 5,716 is a mean absolute error of 0.022 per pixel and 80,138 is 0.306 per pixel.

| Epoch | 2 | 3 | 8 | 9 | 12 | 17 (best) | 19 | 27 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| Validation L1 | 27,937 | 80,138 | 19,008 | 23,482 | 17,253 | 5,716 | 27,094 | 30,298 | 30,478 | 5,940 |

- Eight of 30 epochs are above twice the run median (8,579): epochs 2, 3, 8, 9, 12, 19, 27 and 29. The reported spikes (80,138 early, 30,478 late) are the two largest of a recurring pattern, not two isolated events.
- The training loss falls smoothly from 3.82 to 1.00 and never rises by more than 5% from one epoch to the next. There are 0 non-finite steps. The spikes are therefore visible only in validation.
- Each spike is gone one epoch later (epoch 4: 7,591; epoch 30: 5,940).
- `ReduceLROnPlateau` reads this noisy signal. It halved the learning rate at epoch 28, in this run and in C6 (DINOv2-B, pointwise) only.
- Epoch 17 has the lowest validation L1 and is the saved checkpoint, as intended.

## The same pattern in the other seed-42 runs

Epochs above twice the run median, epoch 1 left out (every perceptual run starts high):

| Run | Spike epochs | Largest validation L1 after epoch 1 | Run median |
|---|---|---|---|
| C10 (CT dose network v2, statistics) | 2, 3, 8, 9, 12, 19, 27, 29 | 80,138 | 8,579 |
| C10 (CT dose network v1, statistics), before the variance fix | 2, 8, 21, 28 | 53,617 | 7,755 |
| C2 (LPIPS-VGG, pointwise) | 6, 13, 15 | 40,860 | 9,802 |
| C7 (MedSigLIP, pointwise) | 7, 28 | 31,891 | 6,933 |
| C4 (RadImageNet, pointwise) | 3 | 26,373 | 7,349 |
| C3 (DISTS, statistics) | 7 | 19,467 | 8,147 |
| C10 (CT dose network v1, statistics), variance fix | none | 11,810 | 6,659 |
| C0 (L1), C1, C5, C6, C8 (before and after the fix), C9 (v1 and v2) | none | below 11,200 | 5,483 to 6,319 |

- Spikes occur in 6 of 15 runs and with four unrelated feature networks. They are not specific to the CT dose network, to statistics matching or to the variance fix (C10 v1 has them before the fix and not after it; C10 v2 has them after it).
- C0 (L1) has none. The spikes come with the perceptual term, which is 95% of the loss at initialisation and leaves the L1 term a small share.
- C10 v2 is the most affected run, in number and in size.

## CPU checks made before the replay

1. **Validation path (code review).** `validation_l1` in `src/train.py` puts the model in eval mode, runs without gradients and without autocast, sums the absolute error and divides by the number of slices. The validation loader is not shuffled. The checkpoint is written only when the score improves. The notebook loop and the seeds notebook call the same function. Nothing here can produce a wrong number for a correct model.
2. **Checkpoint against log (CPU, `python -m src.val_diagnostics`, outputs in `results/val_diagnostics/`).** The saved `best.pth` reproduces the logged best validation L1: C10 v2 5,717.0 against 5,716.1 logged, C0 4,560.1 against 4,560.6 logged. The logged value belongs to the saved weights.
3. **Best checkpoint of C10 v2 against that of C0 (L1), per validation slice.**

| Quantity at the best epoch | C10 v2 (epoch 17) | C0 (epoch 26) |
|---|---|---|
| Validation L1 | 5,717 | 4,560 |
| Median slice | 4,629 | 3,913 |
| Worst slice | 12,528 | 10,482 |
| Share of the total from the 10 worst slices | 11.5% | 11.7% |
| Mean signed error (prediction minus target) | +0.0022 | −0.0012 |
| Share of L1 explained by a per-slice constant offset | 10.4% | 9.7% |
| L1 with BatchNorm on batch statistics (batches of 8) | 10,234 (1.79×) | 7,836 (1.72×) |

At its best epoch C10 v2 looks like C0 with a higher level: no concentration in a few slices, no intensity offset, and the same sensitivity to the BatchNorm mode. The saved model shows no defect. Only C10 v2 and C0 were evaluated (about 18 CPU minutes per checkpoint); C8 and C9 were not.

## GPU replay (2026-10-12)

`DIAGNOSE_C10 = True` in `seeds_perceptual_backbone_ldct.ipynb`, outputs in `results/seed42/diag_C10_v2/` (`replay_check.csv`, `val_per_epoch.csv`, `val_per_slice.csv`). After every epoch it records, per validation slice, the L1, the mean signed error and the L1 with BatchNorm on batch statistics.

**The replay is the same run.** Validation L1 is identical to the committed log at every epoch, the best epoch is 17 and the checkpoint is byte-identical to the original.

**Per epoch.** The L1 with batch statistics has no spikes: it falls steadily from 17,611 to 9,030 and is uncorrelated with the eval-mode L1 after epoch 3 (r = −0.08).

| Epoch | Validation L1 (eval mode) | L1 with batch statistics | Median slice | Share of the 10 worst slices | Mean signed error | Share of L1 from a constant offset |
|---|---|---|---|---|---|---|
| 3 (spike) | 80,138 | 14,146 | 17,323 | 27% | +0.097 | 33% |
| 17 (best) | 5,716 | 10,234 | 4,628 | 11% | +0.002 | 11% |
| 19 (spike) | 27,094 | 10,227 | 5,056 | 36% | +0.078 | 76% |
| 27 (spike) | 30,298 | 9,539 | 5,308 | 31% | +0.094 | 82% |
| 29 (spike) | 30,478 | 9,175 | 5,241 | 33% | +0.093 | 81% |
| 30 | 5,940 | 9,030 | 4,565 | 13% | +0.002 | 11% |

**Per slice.** A slice is called failing when its L1 is above three times its own median over the quiet epochs.

| Spike epoch | Failing slices (of 182) | Patients (of 34) | Their share of the total L1 | Their mean signed error | Mean signed error of the others | Their L1, eval mode | Their L1, batch statistics | L1 of the others |
|---|---|---|---|---|---|---|---|---|
| 2 | 50 | 18 | 84% | +0.274 | +0.004 | 85,501 | 30,753 | 6,133 |
| 3 | 115 | 29 | 95% | +0.154 | +0.001 | 120,397 | 17,028 | 11,037 |
| 8 | 38 | 15 | 73% | +0.231 | +0.008 | 66,487 | 21,158 | 6,480 |
| 9 | 43 | 16 | 83% | +0.073 | +0.005 | 82,056 | 22,253 | 5,362 |
| 12 | 30 | 14 | 62% | +0.214 | +0.010 | 64,862 | 24,483 | 7,857 |
| 19 | 44 | 16 | 86% | +0.323 | 0.000 | 96,909 | 20,964 | 4,834 |
| 27 | 46 | 17 | 87% | +0.363 | +0.003 | 104,141 | 19,813 | 5,321 |
| 29 | 44 | 17 | 86% | +0.376 | +0.004 | 108,414 | 19,182 | 5,628 |

- **The same slices fail each time.** 43 slices fail in at least six of the eight spikes. They come from 17 patients, so this is not one patient.
- **They are the bright slices.** Their mean target intensity is 0.41 to 0.45 against 0.16 to 0.18 for the others (0.27 at epoch 3, where 115 slices fail).
- **The error is a brightness offset.** At epoch 29 their mean prediction is 0.81 against a mean target of 0.43. At the late spikes a constant offset explains about 80% of the total L1.
- **The weights are not the problem on their own.** With batch statistics the same slices have an L1 of about 20,000, three to five times lower than in eval mode.
- **Quiet epochs are not perfectly clean.** Epochs 5, 10, 13 and 15 have 9 to 18 failing slices, below the spike threshold. Epochs 16 to 18 and 28 to 30 except 29 have none.

## Reading

Training runs BatchNorm on batch statistics, where the brightness of a batch is normalised away. Validation runs it on running averages, which are dominated by the darker majority of the training slices. For a bright slice the two modes disagree, and at some epochs the disagreement turns into a strongly positive output offset. The training loss does not see this state: it is computed in batch-statistics mode, and the statistics distance on the CT dose network weighs a constant offset at only 0.43 to 0.61 of white noise at equal MSE (`CLAUDE.md`), with the L1 term at a small share of the loss. This explains why C0 (L1) never spikes and why C10 (CT dose network v2, statistics) spikes most.

Not tested here: whether the spikes in C2, C3, C4, C7 and C10 v1 hit the same slices (their checkpoints at spike epochs do not exist either), and whether a different BatchNorm momentum or normalisation removes them.

## Consequences for the paper

- The C10 row is the epoch-17 checkpoint, chosen by the Protocol rule. It has no failing validation slice and is valid as reported.
- Do not write that training was stable because no step was non-finite. Write that the eval-mode validation L1 of C10 (CT dose network v2, statistics) was unstable (8 of 30 epochs above twice the median), that the cause is a BatchNorm train/eval mismatch on bright slices, and that the checkpoint rule selected epoch 17.
- The Protocol is not changed for one configuration. A different normalisation, BatchNorm momentum or checkpoint rule would need a rerun of every row.
- The learning-rate scheduler reads the eval-mode validation L1, so the spikes also delayed and then triggered the halving at epoch 28. This is part of the Protocol and applies to every row.
- Seeds 123 and 456 will show whether the instability and the early best epoch of C10 repeat. The test split is in eval mode too, so a C10 checkpoint taken at a bad epoch would fail on bright test slices; the validation-based rule is what protects the reported row.
