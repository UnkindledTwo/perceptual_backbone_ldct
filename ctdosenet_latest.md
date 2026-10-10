@UnkindledTwo thanks for the re-screen, very clear tests.

### What we learned
- **Equal block weighting was the wrong fix (my suggestion), and your test shows it.** The scales came from untrained U-Net outputs, where every block gives similar large distances. That says nothing about how well each block ranks noise levels. Blocks 2 and 3 give almost the same distance for 10k, 50k and 100k (0.88, 0.77, 0.73), so rescaling cannot make them grade dose.
- **The screen cannot test the variance bug (H6).** The floor changes distance values by about 1e-4; the damage is to the gradients during training. So "no effect on the screen" does not clear it, and C8 is our headline.
- **The per-block ablation is mostly done already:** the first review found block 1 alone gives M = 0.999. Only the block pairs would be new, so skip it.
- **"Stronger network" (more epochs, augmentation, wider) does not target the cause.** The diagnosed cause is that 6-way cross-entropy saturates: the deeper features learn noisy vs clean, not how noisy. A graded target is the one change aimed at that.

### Plan
- [ ] **First:** commit the variance fix (`(variance.clamp(min=0) + eps).sqrt()`), recalibrate λ, and rerun seed 42 for **C8 and C10** on GPU. This comes before anything else on the dose network.
- [ ] **Dose network, one time-boxed retry, decided by Mon Oct 12:** retrain once with a **log-dose regression (or ordinal) target** and validation checkpointing, with the training loop in `src/`. Judge it on the **validation screen only**.
  - M clearly up (for example above 0.9): it becomes v2, replaces C9/C10, recalibrate λ, keep v1 as a disclosed ablation.
  - Otherwise: freeze v1 and report it honestly ("a task-trained CT network matches grain but not structure"), together with the diagnosis you already have.
  - Skip the accuracy vs M sweep and the wider or deeper variants.
- [ ] **Tue Oct 13 at the latest:** start seeds 123 and 456 for all configurations (about 6.5 GPU hours). The results freeze is Sun Oct 18.
- [ ] Commit the screen scripts from the scratchpad so the tables in this issue can be reproduced.

**GPU:** these tests ran on CPU. Do you still have the A100?

