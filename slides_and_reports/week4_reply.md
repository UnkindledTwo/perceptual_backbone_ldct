## Writing week, day 1: paper set up, Method and Setup drafted, 4-page fit, open questions

2026-10-19. Status after the first writing day, the decisions I had to take to fit 4 pages, and what I need from you. Code and draft: `<commit>` (not committed yet; `paper/`, `index.bib` and the generator changes below are local).

### Done today

- [x] **Fig 3** (from your Writing week comment): seed 42 only; rows image, 96 × 96 px zoom, |prediction − NDCT|. The zoom is chosen by a rule, written in the caption: the window with the median NDCT standard deviation among windows that are at least 90 % soft tissue. Details are in `writingweek_reply.md`.
- [x] **Novelty re-check:** search strings 1 to 7 of `novelty.md` re-run on arXiv. Nothing has been submitted since Sep 28; the verdict stands.
- [x] **Paper set up in `paper/`:**
  - The template is the official ISBI one; it is byte-identical to the zip linked from the 2027 author page.
  - `root.tex` has one file per section.
  - `sync.sh` copies the figures, tables and bibliography in, so the folder also compiles on Overleaf.
  - It compiles locally with Tectonic 0.15.
- [x] **`index.bib`:** 20 entries fetched from doi.org (published versions) or arXiv. It includes the ALDEN (MICCAI 2025) and LEDA (BIBM 2024) venues and the RadImageNet DOI. I added VGG (Simonyan & Zisserman), which the Reference Audit requires but the novelty list was missing.
- [x] **RadImageNet normalization:** the Lab-Rasool model card asks for standard ImageNet normalization, which is what the code uses (`RADIMAGENET_STATS = None`). C4 (RadImageNet, pointwise) was trained with the right input statistics.
- [x] **Method** (2.1 to 2.4) and **Experimental Setup** (dataset, implementation, metrics and statistics) are drafted. Eqs. (1) and (2) were checked against `src/distances.py`. The disclosures are in the text:
  - the threshold lowered from 0.9 to 0.8;
  - the ablation of the first dose network version;
  - the C8 seed-42 run with no recorded code revision;
  - the validation spikes.
- [x] **Every float placed with a takeaway caption:** Fig 1 to 4 and Tables 1 and 2.

### Decisions taken to meet the 4-page limit

The ISBI 2027 rules allow a paid 5th page (references, ethics and acknowledgments only). We will **not** use it: 4 pages including references. With the plan's 2,000 words the draft ran to 6 to 7 pages. Fitting 4 pages needed these changes; none changes a number.

1. **The whole paper in 9 pt** (`\ninept`, an option of the official template). 9 pt is the ISBI minimum, and every text element measures at least that: body, captions, tables, references.
2. **All six floats kept, but compacted:**
   - **Fig 1** redrawn at one-column width (`figures/fig1_overview_column.pdf`, text at 6.5 to 7 pt). The full-width version only had about 4 pt text at column width.
   - **Fig 2** redrawn at one-column width (`figures/fig2_screen_column.pdf`): same two panels, legends below.
   - **Fig 4** at 75 % of the column width.
   - **Table 2** made one-column (`results/week3/table2_compact.tex`): ID, network, matching, parameters, λ, M, B. The timing columns stay in `week3_tables.md`.
   - **Table 1** at 9 pt with the takeaway in its caption (`results/week3/table1_paper.tex`).

   The full-width figures and `table1_final.tex` / `table2_final.tex` are unchanged. A rerun of `src.seeds_tables`, `figures.fig1_overview` and `figures.fig2_screen` reproduces every earlier output byte for byte; the new files are additions.
3. **References:** 17 entries. Author lists longer than six names are cut to "et al." (IEEE style). Skorniewska, Demiroglu and Eulig are not cited, to save space.
4. **Prose budget about 1,500 words instead of 2,000** (Method about 340, Setup about 250, the rest below). The writing plan still says 2,000.

### Where the draft differs from the writing plan

- **Slice-level sentence.** The plan says slice-level tests "agree" with the patient-level ones. They agree on Family A (29 of 30 against 30 of 30) but not on Family B (7 of 9 against 3 of 9). The draft says the slice-level tests are in the repository and does not claim agreement.
- **Screen sentence.** The plan says statistics matching lowers B "for every pretrained network". For MedSigLIP it does not (1.535 pointwise against 1.537 statistics). The Fig 2 and Table 2 captions avoid the claim, and the Results paragraph will name the exception.
- **Abstract length.** The plan says 150 to 200 words; the ISBI template says "about 100 to 150". I will aim for about 130.

### Questions for you

1. **"Checkpoint selection unaffected"** (Implementation; from `writingweek_reply.md`). This was not tested. For C10 (CT dose network v2, statistics) at seed 42 the best epoch has no failing slice, but we do not know that the selected checkpoint would be the same without the spikes. The draft uses your wording for now, marked for discussion. Should it say only that the spikes occur, with the counts per seed?
2. **C8 grain in the zoom** (Fig 3 caption and the Fig 3 and Fig 4 paragraph; also from `writingweek_reply.md`). In the final zoom, C5 (DINOv2-S, pointwise) and C8 (DINOv2-S, statistics) look alike, and it is one slice. The caption uses your wording for now. Should the evidence be the Family B result instead (|1 − R_H| 0.333 to 0.195, patient-level p = 4.9e-4, r = −0.77), with the zoom showing only that L1 removes the grain?
3. **Table 1 marks.** The plan says † and ‡ come from the patient-level tests; `table1_paper.tex` still uses the slice-level marks, and its caption says so. Only one cell changes: C4 (RadImageNet, pointwise) on LPIPS, p = 0.09 at slice level and p = 0.021 at patient level. Should I switch the marks to patient level? That means changing `src/seeds_tables.py` after the freeze, as a new output file.
4. **Word budget.** Do you agree with about 1,500 words and the 9 pt option, or would you rather drop a float? Can the writing plan be updated to match?
5. **Deadline.** The ISBI 2027 page gives Monday Oct 26 and does not mention an extension. Do we plan for Oct 26, as the plan's fallback says, or still Nov 2?
6. **Fang et al.** (withdrawn arXiv paper): cite or not?
7. **AI-use disclosure.** ISBI 2027 requires one in the acknowledgments: the AI system, the sections affected, and the level of involvement. Claude (Anthropic) was used to draft text and to write the bibliography fetch, table generator and figure code. Is there a lab wording for this?

### TODO before the draft on Fri Oct 23

- [ ] **Tue Oct 20:** Results (about 250 words: screen, Table 1, Fig 3 and Fig 4) and Introduction (about 240 words, with Table 1 and the figures referenced in order).
- [ ] **Wed Oct 21:** abstract (about 130 words), Conclusion (about 60), Acknowledgments. ISBI requires funding, a conflict-of-interest statement (even when there is none) and the AI-use disclosure.
- [ ] **Thu Oct 22:** final page fit, the plan's Reference Audit and Figures and Tables Audit, and a script that checks every number in the text against `results/week3/`.
- [ ] **Fri Oct 23:** post the PDF and the open questions here.

Inputs I still need:
- [ ] Author names, order and affiliations (review is single-blind, so they appear in the submission).
- [ ] Funding and grant numbers.
- [ ] The citation for the Poisson low-dose simulation of I2I 2025v1 (it is in the dataset README, not in this repository).
- [ ] The code link: public repository or not.

Smaller items:
- [ ] `index.bib`: Crossref has only the early-access record of DISTS, so its final volume and pages are missing. I did not fill them in from memory.
- [ ] `index.bib`: arXiv lists the D-PerceptCT first author as "Taifour Yousra Nabila"; the name order may be wrong.
- [ ] `index.bib`: MedGemma carries year 2026 from its latest arXiv version (first posted 2025).
- [ ] Commit the work and rerun the figure and table scripts on the clean commit, so `results/week3/manifest.json` records it.
- [ ] Post `week3_freeze_reply.md` and `writingweek_reply.md` here.

### ISBI 2027 compliance (checked today)

| Rule | Status |
|---|---|
| Official template | ✅ identical to the zip on the 2027 page |
| Technical content and references within 4 pages | ✅ 3 pages now; about 300 words of headroom over the remaining budget |
| Letter paper, no page numbers, fonts embedded, no Type 3 fonts | ✅ |
| No font below 9 pt, including captions | ✅ with `\ninept` and 9 pt tables; text inside figures is 5.5 to 7 pt |
| Times font | ✅ TeX Gyre Termes (Times clone) under Tectonic; spconf's Times under pdfLaTeX |
| "Compliance with Ethical Standards" section | ✅ DeepLesion wording from the plan |
| Acknowledgments with funding, conflict of interest and AI-use disclosure | ❌ to do (Wed) |
| Single-blind: author names and affiliations | ❌ to do |
| Submission through EDAS, in-person presentation, at most two first-author submissions per person | to note |

### Reproduce

```bash
python -m src.seeds_tables            # also writes table1_paper.tex and table2_compact.tex (new); older outputs unchanged
python -m figures.fig1_overview       # fig1_overview (unchanged) and fig1_overview_column (new)
python -m figures.fig2_screen         # fig2_screen (unchanged) and fig2_screen_column (new)
python -m figures.fig3_qualitative
cd paper && ./sync.sh && tectonic root.tex
```

<!-- Attach paper/root.pdf (current draft), figures/fig1_overview_column.png and figures/fig3_qualitative.png here. -->
