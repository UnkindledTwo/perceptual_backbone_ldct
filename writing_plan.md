# Writing Plan: Which Feature Network Makes a Good Perceptual Loss for LDCT?

[README](README.md) (research questions, contributions, figure and table list)
· [paper_plan.md](paper_plan.md) (experiment steps and Final Tables)
· [novelty.md](novelty.md) (related work and references)

Week 4 writing, the word budget, the paragraph-by-paragraph checklist
and Week 5 revisions and submission. Numbers come from the
[Final Tables](paper_plan.md#final-tables) in paper_plan.md.

---

## Week 4: Writing (Mon Oct 19 to Sun Oct 25)

**Objective:** a complete 4-page first draft in 4 writing days, shared with
the team on Fri Oct 23, 3 days before the official deadline. No new
experiments this week.

| Day | Writing task |
|---|---|
| Mon Oct 19 | Figures and tables into the LaTeX template; Method and Experimental Setup |
| Tue Oct 20 | Results and Introduction |
| Wed Oct 21 | Abstract, Conclusion, Ethics and Acknowledgments |
| Thu Oct 22 | Page fit to 4 pages, reference audit, final checklist |
| Fri Oct 23 | **Share the first draft with the team** |

- [ ] Copy the lab ISBI template from `projects/_done/proj2proj/isbi/` (`root.tex`
      with one `.tex` file per section, `spconf.sty`, `IEEEbib.bst`,
      `tables/`, `images/`) into `projects/perceptual_backbone_ldct/paper/`;
      clear the old content and point the bibliography to `../index.bib`
- [ ] Create `index.bib` from the Missing list in
      [novelty.md](novelty.md#reference-status) if not done yet
- [ ] Mon Oct 19: re-run the search strings from novelty.md; add anything new
      to its Near Matches and to Introduction P4
- [ ] Mon Oct 19: Figures 1 to 4 and Tables 1 to 2 placed, captions state the
      takeaway; Method and Experimental Setup
- [ ] Tue Oct 20: Results and Introduction
- [ ] Wed Oct 21: Abstract, Conclusion, Ethics and Acknowledgments
- [ ] Thu Oct 22: page fit, Reference Audit and Figures and Tables Audit below
- [ ] **Checkpoint (Fri Oct 23): first draft shared with the team.** Post the
      PDF link and open questions as a Markdown comment on the GitHub issue.
- [ ] Sun Oct 25: team comments collected (weekend review, so the draft can
      be submitted on the official deadline if needed)

### Word Budget

| Section | Target words | Paragraphs | Notes |
|---|---|---|---|
| Title | 8 to 14 | - | research-question form |
| Abstract | 150 to 200 | 1 (6 to 8 sentences) | one number: best family's texture gain over L1 at its PSNR cost, or the screen correlation |
| Index terms | 4 to 5 terms | - | |
| 1 Introduction | 400 to 500 | 5 | Fig 1 on page 1, 3 contributions |
| 2 Method | 600 to 700 | 4 subsections | equations (1) and (2), $\Psi$, screen, dose network |
| 3 Experiments and Results | 500 to 650 | setup + 3 result paragraphs | Table 1, Fig 2, Fig 3, Fig 4 |
| 4 Conclusion | 100 to 150 | 3 short | one limitation |
| Ethics and Acknowledgments | about 40 | 2 | public data template |
| References | 15 to 20 entries | - | |
| **Total prose** | **about 2,000** | | plus about 170 words of captions |

## Final Checklist (Paragraph by Paragraph)

One checkbox per paragraph of the paper. **Tell** is the paragraph's one
message, **Cite** the bibkeys it must contain (details in
[novelty.md](novelty.md#related-work)), **Compare** the numbers it must
contrast (from [Final Tables](paper_plan.md#final-tables)).

### Abstract (150 to 200 words)

- [ ] **S1 Context.** Tell: perceptual losses reduce the over-smoothing of
      LDCT denoisers, and the feature network they use is usually an ImageNet
      VGG chosen by habit.
- [ ] **S2 Gap.** Tell: "However, whether larger, self-supervised or medical
      vision-language networks make better perceptual losses for CT has not
      been tested under one protocol."
- [ ] **S3 Pivot.** Tell: "We compare ten perceptual-loss configurations,
      seven pretrained feature networks from five families and a small CT
      dose network, with pointwise or local statistics matching, as losses for
      one U-Net with matched loss influence and three seeds."
- [ ] **S4 Method.** Tell: a training-free screen (dose ordering, equal-MSE
      blur test) and pointwise vs local statistics matching.
- [ ] **S5 to S6 Result.** Compare: every perceptual loss lowers LPIPS but
      costs PSNR against L1; the best group (VGG-19, LPIPS-VGG, DISTS,
      DINOv2-S statistics) reaches LPIPS of about 0.056 to 0.059 against 0.104
      at a cost of 0.5 to 0.7 dB, with no clear winner inside it. Statistics
      matching lowers the DINOv2-S texture error from 0.333 to 0.195
      (patient-level p = 4.9e-4, r = −0.77, lower at all three seeds) with no
      detectable PSNR or LPIPS difference. The screen does not predict the
      trained outcome. Use the patient-level p-values and effect sizes.
- [ ] **S7 Implication.** Tell: the default VGG is not beaten by larger,
      self-supervised or medical networks; for DINOv2-S, matching local
      statistics instead of features improves texture, but the effect depends
      on the network (for the CT dose network it costs PSNR without a
      significant texture gain); a dose and blur screen cannot replace
      training.

### Index Terms

- [ ] Low-dose CT, denoising, perceptual loss, foundation models, noise
      texture

### 1 Introduction (400 to 500 words)

- [ ] **P1 Motivation (about 80 words).** Tell: pixel losses over-smooth LDCT
      and erase the texture radiologists read; perceptual losses are the
      common fix. Cite: `eulig2024benchmarking`, `yang2018wganvgg`.
- [ ] **P2 Perceptual losses for LDCT (about 100 words).** Tell: VGG is the
      default; CT-specific (SACNN), language-aligned (LEDA) and
      foundation-model (ALDEN, D-PerceptCT) features appear inside individual
      methods, never compared under one protocol. Cite: `li2020sacnn`,
      `chen2024leda`, `wang2025alden`, `taifour2025dperceptct`.
- [ ] **P3 Why bigger is not obviously better (about 100 words).** Tell:
      better classifiers are not better similarity metrics; a large CT quality
      model prefers blurred slices; domain features did not help retinal
      synthesis. Cite: `kumar2022imagenet`, `demiroglu2025medsiglip`,
      `skorniewska2025retinal`, `zhang2018lpips`, `ding2020dists`.
- [ ] **P4 Gap and proposal (about 90 words).** Tell: Viana et al. vary only
      VGG layer and pretraining set and report PSNR and SSIM; we compare
      network families with matched influence, texture and cross-scored
      metrics, and test a screen. Cite: `viana2025perceptual`. Compare: their
      4 VGG settings vs our 11 configurations. No sentence from "Claims we
      must not make" in novelty.md.
- [ ] **P5 Contributions (about 80 words).** Tell: the three contributions
      from the README, stated as found: the controlled comparison shows no
      single best network (Table 1); statistics matching improves texture for
      DINOv2-S at no detectable PSNR cost (Table 1, Family B); the
      training-free screen does not
      predict the trained result (Fig 2, Fig 4).
- [ ] Fig 1 on page 1, referenced in P4 or P5.

### 2 Method (600 to 700 words)

- [ ] **2.1 Objective and feature distances.** Tell: the L1 + $\lambda_\phi
      d_\phi$ objective, equation (1) pointwise and equation (2) local
      statistics, and why random CT noise favours (2). Cite:
      `ding2020dists`, `zhang2018lpips`.
- [ ] **2.2 Feature networks.** Tell: the five families and the chosen layers
      (Table 2), and the shared 224 crop. Cite: `oquab2024dinov2`,
      `sellergren2025medgemma`, `mei2022radimagenet`, `radford2021clip`.
- [ ] **2.3 CT dose network.** Tell: a small CNN (287,328 feature-extractor
      parameters) trained with a log-dose regression target on train-split
      patches, with validation checkpointing, then frozen for every denoiser
      seed; why its features must track noise. Disclose that this v2 replaced
      the 6-class v1 after a validation screen, that the adoption threshold
      was lowered from 0.9 to 0.8 after seeing the screen, and that v1 runs
      are in the supplementary ablation table. Compare: SACNN's autoencoder features.
      Cite: `li2020sacnn`.
- [ ] **2.4 Matched influence and the screen.** Tell: $\lambda_\phi$ from
      $\Psi^\star = 0.95$; the dose ordering $M_\phi$ and equal-MSE blur
      penalty $B_\phi$. Cite: `viana2025perceptual`.
- [ ] Every symbol defined once; only referenced equations numbered.

### 3 Experiments and Results (500 to 650 words)

- [ ] **Dataset.** Tell: I2I 2025v1 from DeepLesion, Poisson simulation at 5
      doses, training at 100k, 899/182/220 slices, patient-level split. Cite:
      `yan2018deeplesion`, the simulation method (TODO bib).
- [ ] **Implementation.** Tell: U-Net, AdamW $10^{-4}$, batch 8, 30 epochs,
      selection on val L1, AMP, GPU, seeds 42, 123 and 456, code link. State
      that the seeds vary denoiser training only; the CT dose network is
      trained once and frozen. One sentence on the eval-mode validation L1
      spikes (checkpoint selection unaffected) and that the seed-42 C8 run
      has no recorded code revision (`results/week3/disclosures.md`). Cite:
      `ronneberger2015unet`.
- [ ] **Metrics and statistics.** Tell: PSNR, SSIM, LPIPS (AlexNet), DISTS,
      $R_H$ with band definition; the circularity marks; Families A and B
      with Holm, paired over the 31 test patients (seeds averaged per slice,
      slices averaged per patient), with effect sizes and seed spread next to
      every p-value. These patient-level tests are the reported results; the
      slice-level tests go to the supplement with one sentence that they
      agree. † marks significantly better and ‡ significantly worse than L1.
      Spearman with permutation test, labelled exploratory,
      noting that some configurations share a feature network (C5 and C8, C9
      and C10, the VGG-based C1 to C3). Cite: `samei2019tg233`.
- [ ] **Screen paragraph (Fig 2).** Tell: every pretrained network orders
      doses almost perfectly ($M_\phi$ 0.95 to 1.00), so ordering does not
      separate them; only the CT dose network has $B_\phi < 1$, that is,
      penalizes blur less than equal-MSE noise. Statistics matching lowers
      $B_\phi$ for every pretrained network but raises it for the CT dose
      network. Compare: MedSigLIP and CLIP vs DINOv2 vs the CT dose network,
      pointwise vs statistics; v1 vs v2 of the CT dose network ($M_\phi$ up,
      trained metrics not uniformly better).
- [ ] **Table 1 paragraph.** Compare: each family vs L1 on every metric with
      deltas and patient-level Holm p; say plainly that every configuration
      loses PSNR (‡ on every PSNR cell, 30 of 30 Family A tests significant);
      name the best LPIPS group and that it is within one seed spread;
      medical pretraining does not help (RadImageNet worst PSNR and the
      smallest significant LPIPS gain, MedSigLIP smoother than L1); the CT dose network matches
      grain statistics but stays far behind on DISTS; no configuration adds
      grain on average ($R_H > 1$ only in single seeds). Mark the direction
      of the texture effect, since MedSigLIP is significantly worse than L1.
- [ ] **Fig 3 and Fig 4 paragraph.** Tell: in the zoom row, DINOv2-S with
      statistics keeps grain closer to NDCT than DINOv2-S pointwise (if the
      final Fig 3 has no zoom row, cite only the error row); the
      screen does not predict the outcome: $B_\phi$ correlates with the
      trained texture error with the opposite sign to the screen's premise,
      probably because statistics configurations have both low $B_\phi$ and
      good texture. No reliable gain from size: DINOv2-B vs -S shows no PSNR
      or LPIPS difference and its texture difference changes sign across
      seeds, so it is not claimed; the parameter-count correlation (ρ = 0.89
      within pointwise configurations) is post hoc and exploratory. Present
      both as consistent with Kumar et al., not as a finding of ours.
      Compare: $\rho$ with and without C2 and C3, and within pointwise
      configurations only (exploratory).

### 4 Conclusion (100 to 150 words)

- [ ] **C1 What we did and found.** Tell: controlled comparison; no network
      is a clear winner, and statistics matching improves texture for
      DINOv2-S.
- [ ] **C2 Recommendation.** Tell: the default ImageNet VGG remains a sound
      choice; try local statistics matching when texture matters, but check
      it per network, since it cost PSNR for the CT dose network; do not
      choose a loss network by a dose-ordering or blur screen alone.
- [ ] **C3 Limitation and next step.** Tell: one simulated dataset, one dose
      in training, one backbone, fixed layers per network, one frozen CT
      network, one run without a recorded code revision; next are real low-dose data and a CT feature network trained
      to detect structural distortion rather than dose.

### Compliance with Ethical Standards

- [ ] "This research study was conducted retrospectively using human subject
      data made available in open access by the NIH Clinical Center
      (DeepLesion). Ethical approval was not required as confirmed by the
      license attached with the open access data."

### Acknowledgments

- [ ] Funding, grant numbers and compute resources (TODO).

### Reference Audit

- [ ] Every compared network cited where it is introduced (DINOv2, MedSigLIP,
      RadImageNet, CLIP, LPIPS, DISTS, VGG)
- [ ] DeepLesion and the Poisson simulation method cited
- [ ] TG-233 NPS cited for $R_H$
- [ ] Viana et al., ALDEN, LEDA, SACNN, D-PerceptCT and Kumar et al. cited and
      differentiated
- [ ] Metadata TODOs from novelty.md resolved (ALDEN and LEDA venues,
      RadImageNet DOI, the withdrawn Fang et al.)
- [ ] 15 to 20 references; published versions preferred over arXiv

### Figures and Tables Audit

- [ ] Fig 1 to Fig 4 and Table 1 to Table 2 referenced in order
- [ ] Captions state the takeaway
- [ ] Same display range and error scale across Fig 3 panels; Fig 3 has one
      seed (42), a zoom row with a rule-chosen soft-tissue crop, and the
      error row
- [ ] Table 1 headers carry ↑/↓ (and →1 for $R_H$), best in bold, mean ± std,
      † significantly better and ‡ significantly worse than L1 (patient-level
      Holm), * for circular metrics
- [ ] Two tables in the paper only; the superseded-runs ablation table
      (`table_ablation_superseded.tex`) goes to the supplement or repository
      and gets one sentence in Method 2.3

---

## Week 5: Revisions and Submission (Mon Oct 26 to Mon Nov 2)

**Objective:** submit on Mon Nov 2, inside the expected extension, or on Mon
Oct 26 if no extension is announced.

| Day | Review milestone |
|---|---|
| Sun Oct 25 | Team comments collected |
| Mon Oct 26 | Official deadline: submit if no extension is announced; otherwise continue |
| Fri Oct 30 | All revisions merged |
| Sun Nov 1 | Final PDF checks (fonts, page limit, figures, references) |
| Mon Nov 2 | **Submit** (confirm deadline time and time zone on the ISBI site) |

- [ ] Mon Oct 26: official deadline. No extension announced: apply the
      weekend comments and submit today. Extension announced: record the new
      deadline in the README and continue.
- [ ] Fri Oct 30: all revisions merged; every comment answered or deferred
      with a reason
- [ ] Sun Nov 1: Submission Compliance below completed
- [ ] **Checkpoint (Mon Nov 2): submitted.** Post the final PDF, submission
      ID and source commit as a Markdown comment on the GitHub issue.

### Submission Compliance (verify against the ISBI 2027 call)

- [ ] Page limit for content and references respected
- [ ] Official template, margins and font sizes unchanged
- [ ] Anonymity rules followed if the call requires them
- [ ] PDF fonts embedded; IEEE PDF compliance check done if required
- [ ] Ethics section present
- [ ] Author list, order and affiliations approved by all authors
- [ ] Final PDF and LaTeX source linked in the GitHub issue
