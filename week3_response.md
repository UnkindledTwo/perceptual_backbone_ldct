## Week 3 review: results, scope check and items before the Oct 18 freeze

@UnkindledTwo Great work: all 33 runs are done four days before the freeze, with clean provenance, integrity checks and frozen λ and dose-network hashes. Below is how the three-seed results answer the scope we agreed on Oct 11 (Opt1 + Opt3), what is still missing from the Week 3 clarification, and how the paper message changes.

### What the three seeds say about the scope

**Opt1, C8 as the practical candidate: weaker than seed 42 suggested.**

| C8 (DINOv2-S, statistics) | Seed 42 (Week 2) | Three seeds (Week 3) |
|---|---|---|
| PSNR | 30.55 | 30.26 ± 0.28 |
| PSNR cost against C0 | 0.33 dB | 0.66 dB |
| LPIPS | 0.0486 | 0.0579 ± 0.0112 (largest spread) |
| Against C5 (DINOv2-S, pointwise) | about +1 dB PSNR, better LPIPS | +0.05 dB median PSNR, LPIPS p = 0.87 |

What survives is the texture effect: C8 against C5 on |1 − R_H| (0.195 against 0.333, r = −0.90). This is the paper's robust positive result. C8 is not tested against C1, C2 or C3, and those four are within one seed spread on LPIPS and DISTS. C8 has the lowest texture error of that group (0.195 against 0.22 to 0.24), but we can only state that descriptively; a new test added now must be labelled exploratory.

**Opt3, the screen's predictive value: answered, and negative.** M sits at 0.95 to 1.00 for every pretrained network, B correlates with the trained texture error with the opposite sign to the premise (ρ = 0.84), and v1 to v2 raised M without better trained metrics. That is a reportable finding. You let the three-seed data decide, as agreed. Keep it exploratory (n = 10, 18 uncorrected rows), and note that B is confounded with the matching rule: statistics matching lowers B for every pretrained network (Table 2 screen), and the statistics configurations also have the best texture.

**Paper message.** The paper is now a benchmark with one clear positive result (statistics matching improves texture for the same network) and one negative result (the screen does not predict the trained outcome), not a "C8 wins" paper. No network is a clear winner, every perceptual loss costs PSNR, and size and medical pretraining do not help. Frame the size result as consistent with Kumar et al., not as our finding (novelty.md, claims we must not make).

### Before the freeze (Sun Oct 18)

- [ ] **Patient-level statistics.** Families A and B are still slice-level Wilcoxon (n = 220). Average seeds per slice, then slices per patient, and run the paired tests over the 31 test patients with Holm as defined. Report effect sizes and seed spread next to every p-value, as you suggested. Expect the small effects (for example C6 against C5, +0.05 dB) to drop out; report whatever comes out.
- [ ] **Direction of the texture mark.** In Table 1, † on |1 − R_H| does not say which side is better (C7 is significantly worse). Use a separate mark or signed deltas.
- [ ] **Screen against outcome.** State that some configurations share a feature network (C5 and C8, C9 and C10, the VGG-based C1 to C3), so the 10 points are not independent networks. The pointwise-only ρ is post hoc: report it as exploratory or leave it out.
- [ ] **Disclosures.** Write down the v2 adoption threshold change (0.9 to 0.8 after seeing the screen), keep v1 C9 and C10 and the pre-fix C8 as a labelled ablation table, and close the open source-revision item for the final C8 run from the Week 2 closeout.
- [ ] **Fig 3.** Export the predictions and build the grid with **C8 next to C5**, C0, C1, C7, C10 (C3 if width allows), same display range and error scale, slices by your median rule.
- [ ] **Fig 1 final, Table 2 final** (287,328 CT feature-extractor parameters, final λ, measured timings, M and B) and the LaTeX export of Tables 1 and 2.
- [ ] Attach Fig 2, Fig 3 and Fig 4 images to the freeze comment (the Week 3 comment still has the placeholder).

### Writing plan

The writing plan (`projects/perceptual_backbone_ldct/writing_plan.md`, now separate from `paper_plan.md`) and the README caption drafts are updated to the observed results: Abstract S3 to S7, P5, Method 2.3 (regression v2 and the disclosures), the statistics, screen, Table 1 and Fig 3/4 paragraphs, and the Conclusion. The planned messages that the results contradict ("no significant PSNR cost", "the screen predicts", "statistics matching raises the blur penalty", "winning family") are replaced. If the patient-level tests change a claim, update the matching Tell line before the writing week starts on Oct 19.

