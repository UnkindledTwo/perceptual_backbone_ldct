

## Review: CT dose network (C9, C10) and next steps

Great work on Week 2, all seed-42 runs done ahead of schedule. The Claude review is solid. Here's what we think is going on and what to try next.

### What's going on
- The CT dose network's **blocks 2 and 3 saturate**: they separate noisy vs clean but can't grade 10k vs 100k. They dominate the summed distance, so C9 lowers its loss 2.5× without improving structure (DISTS barely moves).
- Block 1 orders doses almost perfectly, but it is drowned out by the other blocks.
- The air-crop and resize hypotheses are ruled out. Nice tests.

### Bug (our fault, it's in the plan code)
`local_stats` in `paper_plan.md`: `clamp(min=0).sqrt()` gives NaN gradients, and `clamp(min=1e-8)` avoids that but gives zero gradient in 25–52% of windows. This affects **C8 too**, not just C10.

Fix:
```python
variance = (F.avg_pool2d(feature.pow(2), window) - mean.pow(2)).clamp(min=0)
return mean, (variance + eps).sqrt()
```

### Next steps, in this order, before seeds 123 and 456
- [ ] Apply the variance fix → recalibrate λ → rerun the statistics rows of the screen and seed 42 for **C8 and C10**
- [ ] Dose network v2: **equal per-block weighting** (normalize each block by its distance at initialization). Decide on the **validation screen**, not the test split. One variant only: no block-1-only, no extra tuning
- [ ] v2 **replaces** C9/C10 so the Holm counts stay the same. Keep the old runs as "v1" and disclose them in the text
- [ ] Then start seeds 123 and 456 for all configurations

### Small fixes
- [ ] Table 2: report feature-block parameters only (287k, not 1.18M)
- [ ] Move the dose-network training loop from the notebook into `scripts/` and post a link to the weights
- [ ] Work from the repo version of `paper_plan.md`, not `paper_plan(1).md`

### Heads-up on the story
C8 (DINOv2-S + statistics matching) looks like our headline so far: best LPIPS and DISTS, +1 dB PSNR over the pointwise version (C5). If dose network v2 still loses on DISTS, that's fine; we report it honestly: "a CT-trained network matches grain but not structure."

Also interesting: across the 10 trained networks, a higher blur penalty $B_\phi$ went with **smoother** outputs and lower PSNR (Spearman ρ ≈ +0.88 with $|1-R_H|$, seed 42 only), the opposite of what the plan expected. Keep it in mind for Fig 4.

Please post the results here when done.

