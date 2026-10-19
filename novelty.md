# Novelty Check: Which Feature Network Makes a Good Perceptual Loss for LDCT?

Back to [README](README.md) · Work plan in [paper_plan.md](paper_plan.md)

**Searched on:** 2026-09-28, arXiv API (web search not available in this
session; TODO: repeat S1 and S6 on Google Scholar for MICCAI 2025 and ISBI
2026 proceedings before the writing week). **Re-checked on:** 2026-10-19,
arXiv API, strings 1 to 7 unchanged: no paper submitted since 2026-09-28 (the
known near matches are still returned).

## Search Strings

1. `abs:"perceptual loss" AND (abs:"low-dose" OR abs:LDCT) AND (abs:pretrain* OR abs:backbone OR abs:"feature extractor")`
2. `abs:"perceptual loss" AND (abs:"feature extractor" OR abs:backbone) AND (abs:comparison OR abs:benchmark) AND (abs:denoising OR abs:restoration)`
3. `abs:"perceptual similarity" AND abs:ImageNet AND (abs:accuracy OR abs:scale)`
4. `(abs:"noise level" OR abs:dose) AND abs:"perceptual loss" AND (abs:CT OR abs:denoising) AND (abs:estimat* OR abs:classifier)` (no results)
5. `(abs:DISTS OR abs:"texture similarity") AND abs:loss AND (abs:CT OR abs:medical)` (no relevant results)
6. `(abs:DINOv2 OR abs:CLIP OR abs:"vision transformer") AND abs:"perceptual loss" AND (abs:"super-resolution" OR abs:restoration OR abs:denoising)`
7. `abs:"low-dose CT" AND (abs:DINOv2 OR abs:DINO)` and `abs:"perceptual loss" AND abs:RadImageNet`

Also reused: the near-match reading done on 2026-09-27 for the sibling
`vlm_clean_prior_ssl_ldct` (LEDA, ALDEN, FoundDiff, Transfer CLIP, BioAtt).

## Near Matches

| Paper | Venue, Year | What they actually ran | How we differ | Framing impact |
|---|---|---|---|---|
| Viana, Pereira, Ren, Cavalcanti, Sijbers, "Perceptual Influence: Improving the Perceptual Loss Design for Low-Dose CT Enhancement" (arXiv 2509.23025) | arXiv, 2025 | U-Net on Mayo 2016 (70/10/20 split, one seed). VGG-19 only, pretrained on ImageNet or on a medical dataset, at block3_conv2 or block5_conv4 (4 experiments plus a literature baseline). The loss weight is set so the perceptual term is 95% of the initial loss ("perceptual influence" $\Psi \approx 0.95$). PSNR, SSIM, NRMSE; Wilcoxon on SSIM (Sec. 4 to 6) | One architecture and two pretraining sets; no ViT, VLM, texture-statistics or CT-trained network; no texture or perceptual evaluation; one seed. We reuse their $\Psi$ calibration so our weights are comparable | tighten: closest work; "ImageNet vs medical pretraining matters for LDCT" is theirs |
| Wang, Chen, Song, Fang, Zhang, Tu et al., "Anatomy-Aware Low-Dose CT Denoising via Pretrained Vision Models and Semantic-Guided Contrastive Learning" (ALDEN, arXiv 2508.07788) | MICCAI 2025 (per sibling check; TODO confirm) | DINOv2 or MedSAM (both ViT-B) features inside an anatomy-aware discriminator and an InfoNCE loss; supervised on Mayo 2016 and MCTD; PSNR, SSIM, RMSE, LPIPS, TotalSegmentator Dice; ablation of the two components with DINOv2 | A full GAN plus contrastive method with two foundation models; not a controlled comparison of feature networks as plain perceptual losses; no texture metric, no seeds | tighten: "foundation-model features help LDCT denoising" is theirs |
| Taifour, Beghdadi, Luong, Ming, Zaidi, Cheikh, "D-PerceptCT" (arXiv 2511.14518) | arXiv, 2025 | DINOv2 semantic priors inside the denoising network (ViDex block) plus a contrast-sensitivity perceptual loss (DPRLF); Mayo 2016 | DINOv2 as architecture, not as the loss network | none |
| Chen, Chen, Wang, Niu, Wang, Shan, "Low-dose CT Denoising with Language-engaged Dual-space Alignment" (LEDA, arXiv 2403.06128) | arXiv, 2024 (TODO: venue) | PubMedCLIP-aligned VQGAN trained on NDCT; loss aligns features and text tokens with paired NDCT; RED-CNN and ESAU-Net on Mayo 2016 and 2020 | One language-aligned feature space inside a method; not compared against other feature networks | tighten: a VLM-based loss for LDCT exists |
| Li et al., SACNN (IEEE TMI 2020) (`li2020sacnn`) | IEEE TMI, 2020 | Self-supervised CT autoencoder used as the perceptual loss network for LDCT | Our CT network is trained on a dose-classification task and compared against general networks under one protocol | tighten: "CT-specific perceptual network" is theirs |
| Kumar, Houlsby, Kalchbrenner, Cubuk, "Do better ImageNet classifiers assess perceptual similarity better?" (arXiv 2203.04946) | TMLR, 2022 | Many ImageNet classifiers scored on the BAPPS human-similarity benchmark; better classifiers are not better perceptual metrics beyond a point | Natural images, similarity metric, no training through the network | none; motivates RQ2 |
| Fang, Wang, Arnold, Gong, Pasumarthi, "Medical Foundation Model Features as Perceptual Loss for Brain MRI Contrast Dose Simulation" (arXiv 2608.28773) | arXiv, 2026, **withdrawn** by the authors | RadImageNet, SegVol, BrainIAC vs ImageNet VGG16 and ResNet50 as frozen feature extractors; RadImageNet swapped into an existing MRI dose-simulation GAN | MRI, one swap inside one framework; withdrawn, so cite with care or not at all | none |
| Skorniewska, Papiez, "Exploring the Effectiveness of Deep Features from Domain-Specific Foundation Models in Retinal Image Synthesis" (arXiv 2506.11753) | MIUA 2025 | Retinal foundation-model features as a loss vs perceptual and edge losses for autoencoder synthesis; domain features did not help | Fundus images, synthesis; a counterexample we cite | none; supports "domain is not enough" |
| Lab: Demiroglu, Unal, Ertas, Yildirim, MedSigLIP LDCT IQA (arXiv 2511.12256) and repository `medsiglip-ldct-iqa` README | arXiv, 2025; lab | MedSigLIP fine-tuned on LDCTIQA2023 MOS. The README reports that Gaussian-blurred slices score +0.93 MOS above their originals and that one high-frequency energy scalar reaches 98% of the model's MOS correlation | Evidence for our blur test; we do not use the trained IQA head | none |
| Arda Şahin, tickets #208 and #261 (lab, unpublished) | lab, 2026 | U-Net with L1 + LPIPS ($\lambda$ 0.05 to 0.5, pure), L1 + DISTS ($\lambda$ 0.1 to 0.5, pure, warm-up), DISTS on RadImageNet ResNet50 ($\lambda$ 0.3 to 2), multi-scale LPIPS, DINOv2-S perceptual loss (layers 2, 5, 8, 11); one seed, no stated dose | Our own preliminary work; audited in Step 1.2 | none |

## How to Read This Verdict

Each contribution is compared with its closest prior work on four axes: task,
data type, method or comparison, and evaluation.

- **New:** no prior work matches on method or comparison for this task, or the
  difference is one a reviewer would accept as the contribution.
- **Incremental:** prior work matches on three axes; ours differs in the
  fourth, and that difference is stated as the claim.
- **Already done:** prior work matches on all four axes, or differs only in
  dataset, backbone swap, hyperparameters or seeds. Not a contribution.

| Verdict | Rule | Consequence |
|---|---|---|
| Novel as framed | Every contribution is New | Plan proceeds |
| Novel with tighter framing | At least one New, or at least two Incremental, remain; the Table 1 contribution is among them; claims are reworded | Plan proceeds with the reworded claims below |
| Blocked | The Table 1 contribution is Already done, or no New and fewer than two Incremental remain, or the search was not run | No plan is written until the user picks a pivot or overrides |

## Contribution Check

| Contribution (from the README) | Closest prior work | Task / data / method / evaluation match | What the prior work lacks (checked in its section) | Rating |
|---|---|---|---|---|
| C1: controlled comparison of feature networks from five families (ImageNet CNN, texture-statistics IQA network, self-supervised ViT at two sizes, medical VLM, medical CNN) plus a CT network as perceptual losses for LDCT, with matched perceptual influence, 3 seeds, texture and cross-scored perceptual metrics (Table 1) | Viana et al. 2025 | yes / yes / partly / no | Sec. 4: VGG-19 only, ImageNet vs one medical set, 2 layers; Sec. 5 to 6: PSNR, SSIM, NRMSE only, one seed | New (a new family of compared settings and an evaluation that can change the conclusion) |
| C2: a training-free screen of each feature network (dose ordering across 5 doses and an equal-MSE blur test) and a test of whether it predicts the trained loss quality; answers whether larger or language-supervised networks help (Fig 2, Fig 4) | Kumar et al. 2022 | no / no / partly / no | Natural-image similarity on BAPPS; no training through the network; no noise or dose | New |
| C3: local statistics matching vs pointwise matching for the same network, and a small CT dose-classification network as the feature extractor (Table 1 rows, Table 2) | SACNN (Li et al. 2020); DISTS (Ding et al. 2020) for statistics | yes / yes / partly / partly | SACNN: autoencoder features, pointwise; DISTS: statistics on ImageNet VGG only, designed as a metric | Incremental |

## Verdict

**Novel with tighter framing.** C1 and C2 are New and C1 carries Table 1; C3
is Incremental, so the rule for tighter framing applies. The paper claims a
controlled, texture-aware comparison of feature networks and a screen that
predicts their usefulness; it does not claim that foundation or medical
features are new to LDCT.

**Claims we must not make:**

- "First study of perceptual-loss design for LDCT" (Viana et al. 2025).
- "Medical or domain-specific pretraining changes perceptual losses for LDCT"
  as our finding (Viana et al. 2025; SACNN).
- "First use of DINOv2 or foundation-model features to train an LDCT
  denoiser" (ALDEN, D-PerceptCT).
- "First vision-language loss for LDCT" (LEDA).
- "Better or larger ImageNet models make worse perceptual metrics" as our
  discovery (Kumar et al. 2022); we can say whether it holds for CT losses.
- "A new texture-statistics loss" (DISTS); our statistics variant is a
  simplification applied to other networks.
- "State of the art LDCT denoising"; this is a loss comparison on one U-Net.

**Re-check (done 2026-10-19, strings 1 to 7 on arXiv):** nothing new since
2026-09-28; the verdict and Introduction P4 stay as they are.

## Positioning

| Paper | Feature networks compared | Loss form | Texture or perceptual evaluation | Controlled (one net, matched weight, seeds) | Ours differs by |
|---|---|---|---|---|---|
| Viana et al. 2025 (`viana2025perceptual`) | VGG-19 ImageNet vs medical | pointwise | no | one net, $\Psi$-matched, one seed | 9 networks in 5 families, statistics variant, $R_H$ and cross-scored metrics, 3 seeds |
| ALDEN (`wang2025alden`) | DINOv2 vs MedSAM | discriminator plus InfoNCE | LPIPS, Dice | method, not a controlled comparison | plain perceptual loss, many networks, screen |
| SACNN (`li2020sacnn`) | CT autoencoder | pointwise | no | method | dose-trained CT network, compared with general networks |
| Kumar et al. 2022 (`kumar2022imagenet`) | many ImageNet classifiers | metric only | BAPPS human judgments | not a loss | CT, loss training, dose and blur screen |
| **Ours** | VGG, LPIPS, DISTS, RadImageNet, DINOv2-S/B, MedSigLIP, CT dose network | pointwise and local statistics | PSNR, SSIM, LPIPS, DISTS, $R_H$ | yes, $\Psi$-matched, 3 seeds, Holm | - |

## Metadata Check

- [x] `viana2025perceptual`: arXiv 2509.23025 lists Gabriel A. Viana first.
      The sibling `iqa_loss_for_ldct_reconstruction` still uses the key
      `pereira2025perceptual` (earlier TODO there).
- [ ] TODO: ALDEN (arXiv 2508.07788) has no journal reference on arXiv; confirm
      MICCAI 2025 in the proceedings before citing the venue.
- [ ] TODO: LEDA (arXiv 2403.06128) venue.
- [ ] TODO: Fang et al. (arXiv 2608.28773) is withdrawn; decide with the
      advisor whether to cite it at all.
- [ ] TODO: RadImageNet (Mei et al., Radiology: Artificial Intelligence 2022)
      is not on arXiv; verify the DOI and the source of the PyTorch weights
      Arda used.
- All other near matches match their arXiv records.

## Sibling Overlap

- `iqa_loss_for_ldct_reconstruction` (draft, unassigned): its `ideas.md` Idea B
  ("Medical-Domain Perceptual Loss": ImageNet VGG vs RadImageNet vs
  CT-specific) is this paper. Its Idea A benchmark (8 losses including L1+VGG
  and L1+LPIPS) would share two rows.
- `vlm_clean_prior_ssl_ldct` (Batuhan, ISBI, standard track): a frozen medical
  VLM as a loss in **self-supervised** training, with global embeddings and a
  choice of reference. Shares the MedSigLIP loading code.
- `tissue_band_nps_loss` (Elif, ISBI): a new NPS loss; shares the $R_H$ metric
  code and the U-Net training config.
- `vlm_iqa_analysis` (Tolga, preprint): the trained MedSigLIP IQA model; we use
  its blur finding, not its head.
- **Shared:** U-Net config (AdamW 1e-4, batch 8, 30 epochs, 100k) with
  `vlm_clean_prior_ssl_ldct` and `tissue_band_nps_loss`; $R_H$ band edges with
  `viz_toolbox_roi_robustness`; MedSigLIP loader with
  `vlm_clean_prior_ssl_ldct`.
- **Distinct claims:** this paper is the only one comparing feature networks
  for a **supervised** perceptual loss; Batuhan's asks which **reference** a
  VLM loss needs without clean data; Elif's proposes a spectral loss; Idea A
  compares loss families, not feature networks.
- [ ] TODO (advisor, by Fri Oct 2): mark Idea B of
      `iqa_loss_for_ldct_reconstruction` as moved here, and decide whether the
      Idea A benchmark still runs L1+VGG and L1+LPIPS or reuses this paper's
      runs.

## Related Work

### Perceptual losses for LDCT (Introduction P2 and P4)

- **WGAN-VGG (`yang2018wganvgg`):** VGG perceptual loss for LDCT with a WGAN;
  the default configuration most LDCT papers copy.
- **SACNN (`li2020sacnn`):** CT autoencoder as the perceptual network.
- **Perceptual Influence (`viana2025perceptual`):** VGG layer and pretraining
  set for LDCT; the $\Psi$ weight calibration we reuse.
- **ALDEN (`wang2025alden`):** DINOv2 and MedSAM features in adversarial and
  contrastive losses.
- **LEDA (`chen2024leda`):** language-aligned feature space for LDCT.
- **D-PerceptCT (`taifour2025dperceptct`):** DINOv2 priors in the architecture.

### Perceptual metrics and feature networks (Introduction P3, Section 2)

- **LPIPS (`zhang2018lpips`):** calibrated deep-feature distance; also an
  evaluation metric here.
- **DISTS (`ding2020dists`):** structure and texture statistics of VGG
  features; tolerant to texture resampling.
- **Kumar et al. (`kumar2022imagenet`):** better classifiers are not better
  perceptual metrics.
- **DINOv2 (`oquab2024dinov2`), CLIP (`radford2021clip`), MedSigLIP
  (`sellergren2025medgemma`), RadImageNet (`mei2022radimagenet`):** the
  compared feature networks.
- **Retinal foundation features (`skorniewska2025retinal`):** domain features
  did not help synthesis.

### Datasets, evaluation and foundations (Sections 1 and 3)

- **DeepLesion (`yan2018deeplesion`), U-Net (`ronneberger2015unet`).**
- **NPS (`samei2019tg233`), LDCT benchmarking (`eulig2024benchmarking`).**
- **Lab MedSigLIP IQA (`demiroglu2025medsiglip`):** the blur bias of a large
  CT quality model.

## Reference Status

**Count:** this project has no `index.bib` yet; 21 entries are listed above,
target 15 to 20 after trimming.

**Missing (create `index.bib` before the writing week; copy `yang2018wganvgg`,
`li2020sacnn`, `eulig2024benchmarking` from
`projects/_drafts/iqa_loss_for_ldct_reconstruction/index.bib`):**

- [ ] Viana et al. 2025, arXiv 2509.23025. Needed for: closest work, P4.
- [ ] Wang et al. 2025, ALDEN, arXiv 2508.07788 (venue TODO). Needed for: P2, P4.
- [ ] Chen et al. 2024, LEDA, arXiv 2403.06128. Needed for: P2.
- [ ] Taifour et al. 2025, D-PerceptCT, arXiv 2511.14518. Needed for: P2.
- [ ] Zhang et al. 2018, LPIPS, CVPR, arXiv 1801.03924. Needed for: baseline and metric.
- [ ] Ding et al. 2020, DISTS, IEEE TPAMI, arXiv 2004.07728. Needed for: baseline and metric.
- [ ] Kumar et al. 2022, TMLR, arXiv 2203.04946. Needed for: P3, RQ2.
- [ ] Oquab et al. 2023, DINOv2, arXiv 2304.07193. Needed for: compared network.
- [ ] Radford et al. 2021, CLIP, arXiv 2103.00020. Needed for: screen network.
- [ ] Sellergren et al. 2025, MedGemma Technical Report (MedSigLIP), arXiv 2507.05201. Needed for: compared network.
- [ ] Mei et al. 2022, RadImageNet, Radiology: Artificial Intelligence (DOI TODO). Needed for: compared network.
- [ ] Skorniewska and Papiez 2025, MIUA, arXiv 2506.11753. Needed for: P3.
- [ ] Demiroglu et al. 2025, arXiv 2511.12256. Needed for: blur test motivation.
- [ ] Ronneberger et al. 2015, U-Net, MICCAI. Needed for: backbone.
- [ ] Yan et al. 2018, DeepLesion, J. Med. Imaging. Needed for: dataset.
- [ ] Samei et al. 2019, AAPM TG-233. Needed for: NPS.
- [ ] The Poisson low-dose simulation used by I2I 2025v1 (see the dataset README). Needed for: dataset.

**Drop from the paper:** Fang et al. 2026 unless the advisor wants it
(withdrawn); FoundDiff, Transfer CLIP and BioAtt (architecture or diffusion
methods, not losses).
