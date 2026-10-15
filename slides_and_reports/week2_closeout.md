## Week 2 closeout: experiments complete, final artifacts still need cleanup

@UnkindledTwo I checked the repository at [`db775d5`](https://github.com/UnkindledTwo/perceptual_backbone_ldct/commit/db775d593bbfd97a2ccc068240110b29ac47df74), including the committed per-slice CSVs and training logs. All 11 final seed-42 configurations cover the same 220 test slices and 31 patient IDs. Their metric means agree with the run summaries, and recorded best epochs agree with the minimum validation L1.

The variance fix and `src/dose_net_train.py` are now committed, so those items from the earlier review are complete. C8 after the fix remains the strongest perceptual candidate (30.55 dB, LPIPS 0.0486, DISTS 0.0600). C9/C10 v2 improve dose ordering without uniformly improving downstream quality.

Please complete these items to finalize Week 2:

- [ ] **Regenerate `table1_seed42.csv`, `table2_feature_networks.csv`, and `week1_week2_tables.md`.** The committed summaries still contain pre-fix C8 and v1 C9/C10. Use corrected C8 and regression-v2 C9/C10 for the final rows; retain superseded runs in a clearly labelled ablation table. The spreads here are across test slices, not seeds.
- [ ] **Correct Table 2.** Report 287,328 CT feature-extractor parameters, distinguish total network parameters if included, and use the final lambda values and timings. Do not imply that v2 distance checks were rerun if they were not.
- [ ] **Update reproducibility records.** `environment.json` still names `d3916a2`. Record the source revision actually used for each run, any uncommitted changes at execution time, and exact commands or notebook/cell settings. Add accessible Drive links and hashes for the frozen dose-network and denoiser checkpoints; keep weights out of git.
- [ ] **Resolve the runner inconsistencies before starting the remaining seeds.** In the main notebook, the `DOSE_BLOCK_SCALES` declaration is swallowed by the preceding comment, leaving it undefined. That notebook also still uses the superseded classifier/equal-block-weighting setup. `src/run_config.py` constructs `CTDoseNet()` with six outputs, so it cannot load the one-output regression-v2 checkpoint. The follow-up notebook handles the output count correctly. Make the final entry point unambiguous and check it from a fresh kernel.
- [ ] **Investigate and document C10 validation instability.** The log contains an early validation L1 spike to 80,138, as well as the reported late spike to 30,478. Epoch 17 was correctly selected, but zero non-finite loss steps alone does not establish stable training. Use training/validation diagnostics, not test metrics, to decide whether there is a bug.
- [ ] **Provide the Fig 1 overview draft and update the presentation.** I found no standalone editable overview diagram/final export; the presentation still uses older rows. The diagram should describe the final regression-v2 protocol.
- [ ] **Preserve the adoption-rule disclosure:** the threshold was changed from 0.9 to 0.8 after seeing the validation screen. Reconcile the notebook settings, `decision.json`, and written report rather than presenting this as a prespecified rule.
- [ ] Post the corrected tables, overview draft, artifact links, source provenance, and a short key result/takeaway here, then tick the verified Week 2 deliverables.

**Full paper figures and tables are not expected to be final yet:** the three-seed tables and final Fig 1 to Fig 4 belong to the October 18 Week 3 freeze. Week 2 closeout requires accurate seed-42 artifacts and a reproducible final configuration.

The Week 3 clarification is posted in #298. Keep the current scope: C8 as the practical candidate, and the screen's predictive value as the scientific question; no further network variants.

