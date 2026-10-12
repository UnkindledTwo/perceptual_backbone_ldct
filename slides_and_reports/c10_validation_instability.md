# Validation instability of C10 (CT dose network v2, statistics), seed 42

Written 2026-10-12. Only training and validation data are used here; the test split is not read.

## Verdict

**No bug was found, and the question is not closed.** The committed logs, a review of the validation path and a CPU re-evaluation of the saved checkpoints show that the validation number and the checkpoint choice are correct. They cannot show *why* the validation L1 jumps, because only the best checkpoint of each run was saved. The GPU replay in `seeds_perceptual_backbone_ldct.ipynb` (`DIAGNOSE_C10 = True`, about 10 minutes on the A100) records what is missing. It has not been run yet.

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

## Checks made on 2026-10-12

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

## What is not known

The weights at the spike epochs were not saved, so these three explanations cannot be separated from the existing files:

- **BatchNorm running statistics lag behind the weights.** Training uses batch statistics, validation uses running averages. If the activations move quickly, the eval-mode output can be far off while the train-mode output is fine. The 1.7× gap in the last row above shows that this U-Net is sensitive to the BatchNorm mode even for C0.
- **The weights really pass through bad states.** The statistics distance on the CT dose network barely penalises a constant offset or low-frequency error (0.39 to 0.61 of the white-noise distance at equal MSE, see `CLAUDE.md`), so an intensity drift costs little in the training loss.
- **A few validation slices fail.** Less likely given the size of the mean (0.306 per pixel), but not excluded.

## What the replay records and how to read it

`DIAGNOSE_C10 = True` in `seeds_perceptual_backbone_ldct.ipynb` retrains C10 at seed 42 and writes, per epoch and validation slice, the L1, the mean signed error and the L1 with BatchNorm on batch statistics (`val_per_slice.csv`, `val_per_epoch.csv`). It first compares its validation L1 with the committed log (`replay_check.csv`); if the largest relative difference is not small, the replay is a different run and says nothing about the logged spikes.

| At a spike epoch | Reading |
|---|---|
| `val_l1_batch_statistics` stays near its usual level | Running statistics lag. Not a code bug. The checkpoint rule already avoids these epochs. |
| Both L1 values jump and `share_of_l1_from_offset` is large | The weights drift in intensity: a property of this loss. Report it with C10. |
| `share_of_worst_10_slices` is large | A slice-specific failure. Inspect those slices. |
| The replay does not show the spikes at all | The original run is not reproducible. Treat this as a bug and stop before the seeds. |

## Consequences for the paper

- The C10 row is the epoch-17 checkpoint, chosen by the Protocol rule. It is valid as reported.
- Do not write that training was stable because no step was non-finite. Write that the validation L1 of C10 (CT dose network v2, statistics) was unstable (8 of 30 epochs above twice the median) and that the checkpoint rule selected epoch 17.
- The Protocol is not changed for one configuration. Saving checkpoints at spike epochs, or a different BatchNorm momentum, would need a rerun of every row.
- Seeds 123 and 456 will show whether the instability and the early best epoch of C10 repeat.
