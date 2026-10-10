## Step B: C8 and C10 after the variance fix (seed 42, test split)
| Configuration | Run | lambda | PSNR | SSIM | LPIPS | DISTS | R_H | best epoch | non-finite steps |
|---|---|---|---|---|---|---|---|---|---|
| C8 (DINOv2-S, statistics) | before (clamp 1e-8) | 1.9659 | 30.5308 ± 1.8580 | 0.8660 ± 0.0753 | 0.0483 ± 0.0211 | 0.0602 ± 0.0144 | 0.8573 ± 0.1961 | 30.0000 | 0.0000 |
| C8 (DINOv2-S, statistics) | after (variance fix) | 1.9659 | 30.5473 ± 1.9381 | 0.8666 ± 0.0800 | 0.0486 ± 0.0226 | 0.0600 ± 0.0134 | 0.8605 ± 0.2265 | 27.0000 | 0.0000 |
| C10 (CT dose network v1, statistics) | before (clamp 1e-8): earlier output not found | - | - | - | - | - | - | - | - |
| C10 (CT dose network v1, statistics) | after (variance fix) | 2.9109 | 29.8865 ± 1.8488 | 0.8373 ± 0.0831 | 0.0733 ± 0.0281 | 0.0997 ± 0.0176 | 0.8786 ± 0.2174 | 25.0000 | 0.0000 |

### Lambda
| Configuration | S_1 | S_phi | lambda_new | influence | lambda_old |
|---|---|---|---|---|---|
| C8 (DINOv2-S, statistics) | 0.2740 | 2.6483 | 1.9659 | 0.9500 | 1.9659 |
| C10 (CT dose network v1, statistics) | 0.2740 | 1.7886 | 2.9109 | 0.9500 | 2.9105 |

## Step C: dose network, validation screen and training
| Dose network | Matching | ordering_M | dose_ratio_10k_1000k | blur_penalty_B |
|---|---|---|---|---|
| v1 | pointwise | 0.7242 | 4.9332 | 0.3083 |
| v1 | statistics | 0.7121 | 2.6928 | 0.7542 |
| v2_regression | pointwise | 0.9709 | 4.6044 | 0.3434 |
| v2_regression | statistics | 0.8522 | 2.2845 | 0.8633 |

Decision: ADOPT v2 (rule: M(v2) > 0.8 for pointwise and statistics (validation screen)).

v2 training: validation accuracy 78.71%, clean vs 1000k 92.24%, MAE 0.183 log10 units, best epoch 10 of 10.

## Step D: C9 and C10 on v2 against v1
| Configuration | lambda | PSNR | SSIM | LPIPS | DISTS | R_H |
|---|---|---|---|---|---|---|
| C9 (CT dose network v2 regression, pointwise) | 1.5493 | 30.4038 ± 1.7782 | 0.8583 ± 0.0723 | 0.0676 ± 0.0230 | 0.0972 ± 0.0228 | 0.8527 ± 0.2295 |
| C10 (CT dose network v2 regression, statistics) | 2.8016 | 29.7138 ± 2.0936 | 0.8430 ± 0.1160 | 0.0794 ± 0.0516 | 0.1021 ± 0.0151 | 0.9120 ± 0.2497 |
| C10 (CT dose network v1, statistics), disclosed ablation | - | 29.8865 ± 1.8488 | 0.8373 ± 0.0831 | 0.0733 ± 0.0281 | 0.0997 ± 0.0176 | 0.8786 ± 0.2174 |