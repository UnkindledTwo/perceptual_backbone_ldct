## Writing week: plan updated to the patient-level results

@UnkindledTwo Excellent freeze work in #298: everything is reproducible from `a9aaa30` and every item is closed. The writing week starts Monday (Oct 19), and the writing plan is now updated to your patient-level results: [`writing_plan.md`](https://github.com/itu-biai/dlct-docs/blob/main/projects/perceptual_backbone_ldct/writing_plan.md) (commit `2d79f95`). Please write from it, not from the copied plan in this issue's body.

### What changed in the plan

- **Abstract S5 to S7, P5, Conclusion C1 and C2:** the positive result is now stated with its evidence: statistics matching lowers the DINOv2-S texture error (0.333 to 0.195, patient-level p = 4.9e-4, r = −0.77, lower at all three seeds) with no detectable PSNR or LPIPS difference. It is no longer generalized: for the CT dose network, statistics matching costs 0.54 dB without a significant texture gain, so the recommendation is "try it per network", not "always prefer statistics matching".
- **Metrics and statistics:** the patient-level tests (n = 31) are the reported results; slice-level goes to the supplement with one sentence that they agree. † better and ‡ worse than L1.
- **Table 1 paragraph:** every PSNR cell is ‡ (30 of 30 Family A tests significant); RadImageNet now has the smallest significant LPIPS gain instead of "no gain".
- **Size:** "no reliable gain from size". DINOv2-B vs S on texture is not claimed (sign flips at seed 123, as you noted), and the parameter correlation is post hoc and exploratory, consistent with Kumar et al.
- **Implementation and Limitation:** one sentence on the eval-mode validation spikes and on the seed-42 C8 run without a recorded revision.
- **Two tables only:** the superseded-runs ablation table goes to the supplement or repository, with one sentence in Method 2.3.

### One figure change before Monday

- [ ] **Fig 3:** keep one seed (42) instead of three seed rows, and add a **zoom row with a rule-chosen soft-tissue crop** (for example a fixed-size crop centred on the body centroid, or the crop with the median NDCT local std inside the soft-tissue mask; write the rule in the caption). Rows: image, zoom, |prediction − NDCT|. Same grey range and error scale as now. The seed rows take space without showing anything new, and as you said, grain is invisible at full-slice size. If the zoom is not ready, the Fig 3 text cites the error row only.

### Writing week checklist

- [ ] Mon Oct 19: copy the ISBI template to `projects/perceptual_backbone_ldct/paper/`, place Fig 1 to 4 and the two `.tex` tables, compile once (Overleaf is fine, since the analysis machine has no LaTeX); Method and Experimental Setup
- [ ] Tue Oct 20: Results and Introduction
- [ ] Wed Oct 21: Abstract, Conclusion, Ethics and Acknowledgments
- [ ] Thu Oct 22: page fit to 4 pages, Reference Audit, Figures and Tables Audit
- [ ] **Fri Oct 23: share the first draft here** with the PDF link and open questions

If a sentence you want to write is not backed by a number in `results/week3/`, leave it out or mark it for discussion in the draft.

