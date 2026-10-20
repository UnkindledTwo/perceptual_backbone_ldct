---
name: Perceptual Loss Feature Networks for LDCT
assignee: Arda Şahin
github: UnkindledTwo
status: implementation
venue: isbi2027
paper_type: benchmark
track: standard
draft_due: 2026-10-23
deadline: 2026-11-02
official_deadline: 2026-10-26
issue: https://github.com/itu-biai/dlct_docs/issues/261
---

# Perceptual Loss Feature Networks for LDCT

Which pretrained network should a perceptual loss use when training a
low-dose CT denoiser? This project compares nine feature networks from five
families (ImageNet CNNs, texture-statistics IQA networks, self-supervised
ViTs at two sizes, a medical vision-language model, a medical CNN) and a small
CT dose network, as losses for one U-Net on the lab's I2I 2025v1 data at 100k
photons. It grows out of Arda's LPIPS, DISTS and DINOv2 loss tickets (#208,
#261), Idea B of `iqa_loss_for_ldct_reconstruction`, and idea #2 of
`prj_vlm/notes.md`.

## Results (test split, 220 slices, 100k, seeds 42, 123, 456)

| Configuration | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | R_H (→1) | \|1 − R_H\| ↓ |
|---|---|---|---|---|---|---|
| LDCT input (100k) | 23.04 | 0.7423 | 0.2843 | 0.1641 | 6.174 | 5.275 |
| C0 (L1) | **30.92 ± 0.07** | **0.8691 ± 0.0025** | 0.1043 ± 0.0010 | 0.1112 ± 0.0006 | 0.657 ± 0.026 | 0.352 ± 0.026 |
| C1 (VGG-19, pointwise) | 30.32 ± 0.22‡ | 0.8606 ± 0.0022 | 0.0592 ± 0.0010† | 0.0668 ± 0.0023 | 0.798 ± 0.035 | 0.226 ± 0.030† |
| C2 (LPIPS-VGG, pointwise) | 30.39 ± 0.31‡ | 0.8630 ± 0.0021 | 0.0563 ± 0.0045†* | **0.0645 ± 0.0024** | 0.777 ± 0.012 | 0.242 ± 0.009† |
| C3 (DISTS, statistics) | 29.81 ± 0.20‡ | 0.8326 ± 0.0244 | **0.0557 ± 0.0066**† | 0.0664 ± 0.0014* | 0.820 ± 0.014 | 0.222 ± 0.018† |
| C4 (RadImageNet, pointwise) | 28.70 ± 0.32‡ | 0.8259 ± 0.0075 | 0.0868 ± 0.0055† | 0.0847 ± 0.0009 | 0.798 ± 0.040 | 0.286 ± 0.017† |
| C5 (DINOv2-S, pointwise) | 30.10 ± 0.51‡ | 0.8625 ± 0.0063 | 0.0629 ± 0.0078† | 0.0707 ± 0.0068 | 0.682 ± 0.046 | 0.333 ± 0.044† |
| C6 (DINOv2-B, pointwise) | 30.20 ± 0.44‡ | 0.8618 ± 0.0044 | 0.0614 ± 0.0027† | 0.0713 ± 0.0028 | 0.731 ± 0.027 | 0.291 ± 0.029† |
| C7 (MedSigLIP, pointwise) | 29.84 ± 0.39‡ | 0.8596 ± 0.0061 | 0.0750 ± 0.0114† | 0.0737 ± 0.0070 | 0.531 ± 0.063 | 0.477 ± 0.060‡ |
| C8 (DINOv2-S, statistics) | 30.26 ± 0.28‡ | 0.8614 ± 0.0068 | 0.0579 ± 0.0112† | 0.0654 ± 0.0047 | 0.857 ± 0.017 | 0.195 ± 0.013† |
| C9 (CT dose network v2, pointwise) | 30.44 ± 0.10‡ | 0.8514 ± 0.0219 | 0.0673 ± 0.0031† | 0.0973 ± 0.0002 | 0.850 ± 0.018 | 0.203 ± 0.010† |
| C10 (CT dose network v2, statistics) | 29.80 ± 0.19‡ | 0.8475 ± 0.0039 | 0.0721 ± 0.0063† | 0.0982 ± 0.0036 | 0.951 ± 0.058 | **0.167 ± 0.026**† |

Mean ± standard deviation over the three per-seed means. Bold: best trained configuration. † significantly better and
‡ significantly worse than C0 (L1) (Holm-adjusted p < 0.05, paired Wilcoxon test on the 31 patient means of
seed-averaged per-slice values; tested on PSNR, LPIPS and |1 − R_H|). * scored with its own training metric. Table 1 marks
are patient level (decided 2026-10-20; `results/week3/table1_final_patient.csv`, `table1_paper_patient.tex`). At slice level
only one mark differs: C4 (RadImageNet, pointwise) on LPIPS is not significant (Holm p = 0.09, against 0.021 per patient);
the frozen `table1_final.csv` and `.tex` keep the slice-level marks. All tests (per slice and per patient), signed differences to C0, the screen-versus-outcome correlations, Table 2, the ablation of
superseded runs and the validation-spike counts are in `results/week3/week3_tables.md`; disclosures in
`results/week3/disclosures.md`; LaTeX tables in `results/week3/*.tex`. Freeze commit: `a9aaa30` (2026-10-15); Fig 3 was rebuilt with one seed and a zoom row on 2026-10-19.

## ISBI 2027 Plan

| File | What it is for | Who reads it when |
|---|---|---|
| [paper_plan.md](paper_plan.md) | Step-by-step work plan: tasks, formulas, code, result tables | Student, daily from Week 1 to submission |
| [writing_plan.md](writing_plan.md) | Writing week, word budget, paragraph-by-paragraph checklist, revisions and submission | Student, from the writing week to submission |
| [novelty.md](novelty.md) | Search log, near matches, verdict, related work, sibling overlap, missing references | Advisor at plan review; student on the Introduction writing day |

### Timeline

| Week | Dates | Focus | Milestone |
|---|---|---|---|
| Plan phase | on or before Mon Sep 28 | Scoping | RQs frozen, novelty verified, plan approved |
| Week 1 | Mon Sep 28 to Sun Oct 4 | Data, reference and the training-free screen | C0 row, audit of earlier runs, Fig 2 draft |
| Week 2 | Mon Oct 5 to Sun Oct 11 | Perceptual losses and the CT dose network | C1 to C10 (seed 42), Table 2, Fig 1 draft |
| Week 3 | Mon Oct 12 to Sun Oct 18 | Seeds, statistics and figures | **Results freeze** (Sun Oct 18) |
| Week 4 | Mon Oct 19 to Sun Oct 25 | Writing (Mon Oct 19 to Thu Oct 22) and first team comments | **First draft shared with the team (Fri Oct 23)** |
| Week 5 | Mon Oct 26 to Mon Nov 2 | Revisions and Submission | **Submission (Mon Nov 2)** |
| - | Mon Oct 26 | Official ISBI deadline | Submit now if no extension is announced (see fallback) |

**Fallback if ISBI announces no extension by Mon Oct 26:** apply the weekend
team comments and submit the first draft on Mon Oct 26. The planned
submission on Mon Nov 2 assumes the usual extension.

**Why the standard track:** Arda's earlier runs cover LPIPS, DISTS and
DINOv2, but single seed, unknown dose and an unexplained SSIM drop; they are
audited in Week 1 and treated as a head start, not as final rows.

### Research Focus

**Working title:** Which Feature Network Makes a Good Perceptual Loss for
Low-Dose CT Denoising?

**Core question:** Do larger, self-supervised or medical vision-language
feature networks make better perceptual losses for LDCT denoising than the
default ImageNet VGG, and can a cheap test run before training tell which
network will work?

**Key insight:** a loss network must change its features when noise changes
and must penalize blur at least as much as noise. Large quality models do not
guarantee this: the lab's MedSigLIP LDCT quality model scores Gaussian-blurred
slices higher than their originals (`medsiglip-ldct-iqa` README). Because CT
noise is random, matching local feature statistics rather than features pixel
by pixel should keep texture without forcing the output to copy the
reference grain.

**Protocol in one equation:**

$$
\mathcal{L}_\phi(\hat{x}, x) = \lVert \hat{x} - x \rVert_1 + \lambda_\phi \, d_\phi\big(\mathrm{crop}(\hat{x}), \mathrm{crop}(x)\big), \qquad \frac{\lambda_\phi S_\phi}{S_1 + \lambda_\phi S_\phi} = 0.95
$$

where $\hat{x}$ is the U-Net output, $x$ the normal-dose target, $d_\phi$ the
pointwise or local-statistics distance in the features of network $\phi$,
$\mathrm{crop}$ a shared random 224 × 224 crop, and $\lambda_\phi$ set so the
perceptual term has the same share (0.95) of the initial loss for every
network, following Viana et al. 2025 ($S_1$, $S_\phi$ are the summed L1 and
feature terms at initialization).

**Research questions:**

1. With one U-Net, one budget and matched loss influence, which feature
   network family gives the best texture and perceptual quality at acceptable
   PSNR? Answered by [Table 1](paper_plan.md#final-tables).
2. Does a training-free screen (dose ordering and an equal-MSE blur test)
   predict how good a loss a network makes, and does network size or
   language supervision help? Answered by Fig 2
   ([Step 1.4](paper_plan.md#step-14-training-free-screen-dose-ordering-and-the-equal-mse-blur-test))
   and Fig 4 ([Step 3.2](paper_plan.md#step-32-statistics-and-the-screen-vs-outcome-test)).
3. Does local statistics matching beat pointwise matching for the same
   network, and can a small CT dose network compete with large pretrained
   ones? Answered by Table 1 rows C5, C8, C9, C10 and the Family B tests
   ([Step 3.2](paper_plan.md#step-32-statistics-and-the-screen-vs-outcome-test)).

**Contributions:**

1. A controlled comparison of feature networks from five families and a CT
   network as perceptual losses for LDCT, with matched loss influence, 3
   seeds, Holm-corrected tests, and texture ($R_H$) plus cross-scored
   perceptual metrics. Evidence: Table 1.
2. A training-free screen of loss networks (dose ordering, equal-MSE blur
   penalty) and a test of whether it predicts the trained result, including
   the size and vision-language axes. Evidence: Fig 2, Fig 4.
3. Local statistics matching as an alternative to pointwise matching, and a
   small CT dose-classification network as the feature extractor. Evidence:
   Table 1 (C8 to C10), Table 2.

**What this paper is NOT:** not a new denoiser architecture; not a GAN or
contrastive method (that is ALDEN); no VLM conditioning or text prompts (that
is `text_conditioned_ldct_denoising`); no self-supervised training (that is
`vlm_clean_prior_ssl_ldct`); no trained IQA model as a reward
(`vlm_iqa_analysis`, `iqa_rewarded_diffusion`); no layer search per network
(Viana et al. did it for VGG); no mixed losses such as DINOv2 + LPIPS; no
SwinIR or other backbones; one dataset and one training dose; no reader
study.

### Novelty Verdict

**Novel with tighter framing** (C1 and C2 New, C3 Incremental). The closest
work, Viana et al. 2025 (arXiv 2509.23025), varies only VGG-19 layer and
pretraining set for LDCT and reports PSNR, SSIM and NRMSE; ALDEN (MICCAI 2025)
and D-PerceptCT use DINOv2 inside methods rather than comparing loss
networks. The paper must not claim the first perceptual-loss study for LDCT,
the first foundation-model or VLM loss for LDCT, or the "better classifier,
worse metric" finding. Details, metadata checks and sibling overlap in
[novelty.md](novelty.md).

### Figures and Tables

| ID | Content | Width | Made in | Script or source | Caption draft (with takeaway) | Status |
|---|---|---|---|---|---|---|
| Fig 1 | Overview: feature zoo grouped by family, the training-free screen, $\Psi$-matched U-Net training, evaluation | 1 col | [Step 2.3](paper_plan.md#step-23-seed-42-runs-for-c1-to-c10-and-the-fig-1-draft) draft, [Step 3.3](paper_plan.md#step-33-figures-and-table-export) final | `figures/fig1_overview.py` | "Only the feature network in the loss changes; its weight is matched by influence, and a screen scores each network before training." | done |
| Fig 2 | Screen: (a) feature distance vs dose, one line per network; (b) equal-MSE blur penalty per network, pointwise vs statistics | 1 col | [Step 1.4](paper_plan.md#step-14-training-free-screen-dose-ordering-and-the-equal-mse-blur-test) data, [Step 3.3](paper_plan.md#step-33-figures-and-table-export) plot | `figures/fig2_screen.py` | "Every pretrained network orders doses almost perfectly; only the CT dose network penalizes blur less than noise of the same pixel error, and statistics matching lowers the blur penalty of every pretrained network." | done |
| Fig 3 | Qualitative grid, seed 42: LDCT, C0, C1, C3, C5, C7, C8, C10, NDCT; image row, zoom row with a rule-chosen soft-tissue crop, error row; slice chosen by rule (median C0 PSNR) | 2 col | [Step 3.3](paper_plan.md#step-33-figures-and-table-export) | `figures/fig3_qualitative.py` | "L1 smooths the grain away; statistics matching keeps DINOv2-S grain closer to NDCT than pointwise matching does." | done |
| Fig 4 | Blur penalty vs trained texture error, marker size by parameters, colour by family, DINOv2 S and B joined | 1 col | [Step 3.2](paper_plan.md#step-32-statistics-and-the-screen-vs-outcome-test) data, [Step 3.3](paper_plan.md#step-33-figures-and-table-export) plot | `figures/fig4_screen_vs_outcome.py` | "The screen does not predict the trained result: the blur penalty rises with the trained texture error (ρ = 0.84, exploratory), and larger networks do not give better texture." | done |
| Table 1 | Main results: LDCT input and C0 to C10 × PSNR, SSIM, LPIPS, DISTS, $R_H$; mean ± std; † Holm p < 0.05 vs L1; * circular metric | 2 col | Rows from [Step 1.2](paper_plan.md#step-12-audit-earlier-runs-and-train-the-l1-reference) and [Step 2.3](paper_plan.md#step-23-seed-42-runs-for-c1-to-c10-and-the-fig-1-draft), final in [Final Tables](paper_plan.md#final-tables) | `src/seeds_tables.py` (`results/week3/table1_final_patient.csv`, `table1_paper_patient.tex`; † / ‡ patient level) | "Every perceptual loss costs PSNR; the best perceptual group is within one seed spread, statistics matching gives the best texture for DINOv2-S, and medical pretraining does not help." | done |
| Table 2 | Feature networks: family, parameters, pretraining, layers, input, $\lambda_\phi$, ms per step, screen scores | 1 col | [Step 2.1](paper_plan.md#step-21-calibrate-each-loss-weight-by-perceptual-influence), [Step 1.4](paper_plan.md#step-14-training-free-screen-dose-ordering-and-the-equal-mse-blur-test), [Step 2.2](paper_plan.md#step-22-train-the-ct-dose-network) | `src/seeds_tables.py` (`results/week3/table2_final.csv`, `.tex`) | - | done |

### Decisions and Open TODOs

- [ ] Advisor, by Fri Oct 2: mark Idea B of `iqa_loss_for_ldct_reconstruction`
      as moved here, and decide whether its Idea A benchmark reuses this
      paper's VGG and LPIPS runs
- [x] Dedicated GitHub issues for the ISBI plan (itu-biai/dlct-docs#296 to
      #300, one per week)
- [x] Hugging Face access to `google/medsiglip-448` (C7 trained at all three
      seeds)
- [x] Normalization of the RadImageNet ResNet-50 weights (loaded from
      `Lab-Rasool/RadImageNet` on Hugging Face): the model card asks for
      standard ImageNet normalization, which is what the runs use
      (`RADIMAGENET_STATS = None`)
- [ ] The Step 1.2 audit table of Arda's #208 and #261 runs is not in the
      outputs; the SSIM drop of those runs is not explained (the runs here use
      `data_range=1.0`, and every output is above the input SSIM of 0.742)
- [x] The U-Net is `src/model.py`
- [ ] The $R_H$ band edges from `viz_toolbox_roi_robustness` (the code uses
      0.25 to 0.5 cycles per pixel)
- [ ] Metadata: ALDEN and LEDA venues, RadImageNet DOI, whether to cite the
      withdrawn Fang et al. 2026 ([novelty.md](novelty.md#metadata-check))
- [ ] Create `index.bib` from the 17 missing references in novelty.md before
      the writing week

## Rebuilding the Week 3 tables and figures (CPU, no data)

```
python -m venv ~/.venvs/ldct-analysis && ~/.venvs/ldct-analysis/bin/pip install -r requirements-analysis.txt
python -m src.seeds_tables            # results/week3/*.csv, *.tex, week3_tables.md, disclosures.md, provenance_seeds.json
python -m figures.fig1_overview       # figures/fig1_overview.{pdf,svg,png}
python -m figures.fig2_screen         # figures/fig2_screen.{pdf,svg,png}
python -m figures.fig4_screen_vs_outcome
python -m figures.fig3_qualitative    # needs results/week3/fig3/ from the EXPORT_FIG3 cell of the seeds notebook (Colab)
python -m src.record results/week3/results_before_week3.sha256   # older result files unchanged?
```

Every command writes its inputs, outputs and their sha256, the package versions and the source revision to
`results/week3/manifest.json`. Outputs are byte-identical on a rerun. Nothing outside `results/week3/` and
`figures/` is written; `python -m src.tables` (Weeks 1 and 2, seed 42) rewrites its own older files and is not part
of this step.

## Layout

- `seeds_perceptual_backbone_ldct.ipynb`: the only entry point for training runs (seeds 123 and 456, A100).
- `src/`: modules (`seeds_tables.py`, `stats.py`, `record.py`, `latex.py` for Week 3; `tables.py` for the seed-42 tables).
- `results/`: per-run metrics, logs and summaries (`seed42/`, `seed123/`, `seed456/`), `week3/` for the three-seed analysis,
  `provenance.json` and `week3/provenance_seeds.json` for the source revision and checkpoint hash of every run.
- `figures/`: figure scripts and their outputs.
- `slides_and_reports/`: slide decks, CT dose network reports and the weekly GitHub issue comments.
- Model weights (`*.pt`, `*.pth`) are not in the repository.
