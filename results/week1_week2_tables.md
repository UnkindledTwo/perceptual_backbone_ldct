### Environment

| Item | Value |
|---|---|
| python | 3.13.15 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| cuda | 13.0 |
| gpu | NVIDIA A100-SXM4-80GB |
| gpu_memory_gb | 79.3000 |
| cudnn_deterministic | yes |
| repo_commit | d3916a2af8701cca2aad813d96a1f52ada9f4580 |
| seed | 42 |
| epochs | 30 |
| batch_size | 8 |
| crop_size | 224 |
| target_influence | 0.9500 |

### Step 1.1 checks

| Check | Expected | Observed |
|---|---|---|
| Train / val / test patients | disjoint | 115 / 34 / 31 |
| Train / val / test slices | 899 / 182 / 220 | 899 / 182 / 220 |
| Shared patients between splits | 0 | 0 |
| Input and target range | [0, 1] | [0.000, 1.000] |
| Metrics on identity input | PSNR inf, SSIM 1, LPIPS 0, DISTS 0, R_H 1 | PSNR inf, SSIM 1, LPIPS 0, DISTS 1.19e-07, R_H 1 |
| SSIM of noisy_clipped_100k vs gt_clipped, test mean | below the U-Net outputs | 0.7423 |

### Step 1.3 distance checks

| Network | Matching | distance_at_equality | gradient_reaches_prediction | ms_per_batch_of_8 |
|---|---|---|---|---|
| VGG-19 relu3_2 | pointwise | 0.0000 | yes | 11.1039 |
| VGG-19 relu3_2 | statistics | 0.0000 | yes | 11.9008 |
| LPIPS-VGG | pointwise | 0.0000 | yes | 27.0968 |
| DISTS | statistics | 0.0000 | yes | 26.3683 |
| RadImageNet ResNet-50 | pointwise | 0.0000 | yes | 20.6318 |
| RadImageNet ResNet-50 | statistics | 0.0000 | yes | 22.7277 |
| DINOv2-S | pointwise | 0.0000 | yes | 31.4855 |
| DINOv2-S | statistics | 0.0000 | yes | 28.8735 |
| DINOv2-B | pointwise | 0.0000 | yes | 79.4489 |
| DINOv2-B | statistics | 0.0000 | yes | 80.0561 |
| DINOv2-L | pointwise | 0.0000 | yes | 250.1011 |
| DINOv2-L | statistics | 0.0000 | yes | 251.4793 |
| CLIP ViT-B/16 | pointwise | 0.0000 | yes | 59.6873 |
| CLIP ViT-B/16 | statistics | 0.0000 | yes | 58.9357 |
| MedSigLIP-448 | pointwise | 0.0000 | yes | 1370.4594 |
| MedSigLIP-448 | statistics | 0.0000 | yes | 1372.8979 |
| CT dose network | pointwise | 0.0000 | yes | 8.7048 |
| CT dose network | statistics | 0.0000 | yes | 11.5900 |

### Step 1.4 training-free screen (validation split)

| Network | Matching | ordering_M | dose_ratio_10k_1000k | blur_penalty_B | Parameters |
|---|---|---|---|---|---|
| Pixel L1 (reference) | - | 1.0000 | 5.7367 | 0.8861 | 0 |
| VGG-19 relu3_2 | pointwise | 1.0000 | 2.6057 | 1.5092 | 1145408 |
| VGG-19 relu3_2 | statistics | 0.9989 | 1.6763 | 1.4181 | 1145408 |
| LPIPS-VGG | pointwise | 0.9984 | 2.4252 | 1.4825 | 14714688 |
| DISTS | statistics | 0.9835 | 2.7509 | 1.4495 | 14714688 |
| RadImageNet ResNet-50 | pointwise | 0.9538 | 7.1007 | 1.8118 | 25557032 |
| RadImageNet ResNet-50 | statistics | 0.9330 | 3.5960 | 1.3961 | 25557032 |
| DINOv2-S | pointwise | 0.9978 | 7.7117 | 1.7518 | 22056576 |
| DINOv2-S | statistics | 0.9973 | 2.3666 | 1.2604 | 22056576 |
| DINOv2-B | pointwise | 0.9973 | 6.2713 | 1.6694 | 86580480 |
| DINOv2-B | statistics | 0.9940 | 2.3098 | 1.2501 | 86580480 |
| DINOv2-L | pointwise | 0.9956 | 6.8865 | 1.8737 | 304368640 |
| DINOv2-L | statistics | 0.9967 | 2.3219 | 1.2776 | 304368640 |
| CLIP ViT-B/16 | pointwise | 0.9956 | 2.7337 | 1.5710 | 85799424 |
| CLIP ViT-B/16 | statistics | 0.9901 | 1.7508 | 1.4193 | 85799424 |
| MedSigLIP-448 | pointwise | 0.9995 | 3.2543 | 1.5346 | 428565440 |
| MedSigLIP-448 | statistics | 0.9956 | 2.3334 | 1.5369 | 428565440 |
| CT dose network | pointwise | 0.7242 | 4.9332 | 0.3083 | 1175142 |
| CT dose network | statistics | 0.7121 | 2.6929 | 0.7544 | 1175142 |

### Step 2.1 calibration

| Configuration | Label | S_1 | S_phi | lambda |
|---|---|---|---|---|
| C0 | L1 | 0.2740 | - | 0.0000 |
| C1 | VGG-19 | 0.2740 | 1.1080 | 4.6992 |
| C2 | LPIPS-VGG | 0.2740 | 0.5682 | 9.1625 |
| C3 | DISTS | 0.2740 | 0.7455 | 6.9836 |
| C4 | RadImageNet | 0.2740 | 2.3552 | 2.2106 |
| C5 | DINOv2-S pointwise | 0.2740 | 3.1857 | 1.6343 |
| C6 | DINOv2-B pointwise | 0.2740 | 2.0849 | 2.4972 |
| C7 | MedSigLIP | 0.2740 | 1.3864 | 3.7553 |
| C8 | DINOv2-S statistics | 0.2740 | 2.6484 | 1.9659 |
| C9 | CT dose net pointwise | 0.2740 | 3.2743 | 1.5901 |
| C10 | CT dose net statistics | 0.2740 | 1.7889 | 2.9105 |

### Step 2.2 CT dose network checks

| Check | Observed |
|---|---|
| Val accuracy, 6 classes | 78.48% |
| Val accuracy, clean vs 1000k | 90.52% |
| Screen ordering M_phi (pointwise / statistics) | 0.7242 / 0.7121 |
| Screen blur penalty B_phi (pointwise / statistics) | 0.3083 / 0.7544 |
| Mean body fraction of a training patch | 0.987 |

### Step 2.2 CT dose network, per class

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| 10k | 0.9928 | 0.9505 | 0.9712 | 728 |
| 50k | 0.9038 | 0.7610 | 0.8262 | 728 |
| 100k | 0.7741 | 0.7720 | 0.7730 | 728 |
| 500k | 0.5954 | 0.5742 | 0.5846 | 728 |
| 1000k | 0.6328 | 0.7527 | 0.6876 | 728 |
| clean | 0.8560 | 0.8984 | 0.8767 | 728 |

### Step 1.2 and Step 2.3: Table 1 rows, seed 42 (spread over test slices)

| Configuration (seed 42, test, 100k) | PSNR (dB) | SSIM | LPIPS | DISTS | R_H | ms per step |
|---|---|---|---|---|---|---|
| LDCT input (100k) | 23.04 ± 2.13 | 0.7423 ± 0.0647 | 0.2843 ± 0.0959 | 0.1641 ± 0.0462 | 6.1743 ± 5.8560 | - |
| C0 L1 | 30.88 ± 1.97 | 0.8715 ± 0.0946 | 0.1036 ± 0.0704 | 0.1115 ± 0.0203 | 0.6522 ± 0.2270 | 118 |
| C1 VGG-19 | 30.17 ± 1.94 | 0.8619 ± 0.0905 | 0.0599 ± 0.0257 | 0.0694 ± 0.0166 | 0.7689 ± 0.1832 | 129 |
| C2 LPIPS-VGG | 30.73 ± 1.96 | 0.8639 ± 0.0766 | 0.0570 ± 0.0210 | 0.0627 ± 0.0137 | 0.7841 ± 0.1763 | 141 |
| C3 DISTS | 30.03 ± 1.76 | 0.8045 ± 0.0573 | 0.0513 ± 0.0204 | 0.0649 ± 0.0129 | 0.8360 ± 0.1901 | 147 |
| C4 RadImageNet | 28.85 ± 1.51 | 0.8342 ± 0.0876 | 0.0832 ± 0.0308 | 0.0841 ± 0.0161 | 0.7647 ± 0.2506 | 134 |
| C5 DINOv2-S pointwise | 29.53 ± 2.07 | 0.8649 ± 0.0803 | 0.0595 ± 0.0305 | 0.0673 ± 0.0133 | 0.6549 ± 0.2163 | 150 |
| C6 DINOv2-B pointwise | 29.69 ± 1.78 | 0.8567 ± 0.0846 | 0.0639 ± 0.0372 | 0.0745 ± 0.0143 | 0.7156 ± 0.2707 | 201 |
| C7 MedSigLIP | 29.90 ± 1.86 | 0.8575 ± 0.0969 | 0.0701 ± 0.0477 | 0.0706 ± 0.0140 | 0.5822 ± 0.2160 | 1485 |
| C8 DINOv2-S statistics | 30.53 ± 1.86 | 0.8660 ± 0.0753 | 0.0483 ± 0.0211 | 0.0602 ± 0.0144 | 0.8573 ± 0.1961 | 153 |
| C9 CT dose net pointwise | 30.26 ± 1.80 | 0.8456 ± 0.0718 | 0.0781 ± 0.0255 | 0.1001 ± 0.0229 | 0.8709 ± 0.1925 | 128 |
| C10 CT dose net statistics | 30.17 ± 2.04 | 0.8544 ± 0.0918 | 0.0676 ± 0.0296 | 0.0934 ± 0.0171 | 0.8753 ± 0.2446 | 132 |

### Table 2 values per trained network

| Configuration | Network | Matching | Parameters | lambda | ms_per_step | ms_per_batch_of_8 | M_phi | B_phi |
|---|---|---|---|---|---|---|---|---|
| C1 | VGG-19 relu3_2 | pointwise | 1145408 | 4.6992 | 129.3513 | 11.1039 | 1.0000 | 1.5092 |
| C2 | LPIPS-VGG | pointwise | 14714688 | 9.1625 | 140.8097 | 27.0968 | 0.9984 | 1.4825 |
| C3 | DISTS | statistics | 14714688 | 6.9836 | 146.9189 | 26.3683 | 0.9835 | 1.4495 |
| C4 | RadImageNet ResNet-50 | pointwise | 25557032 | 2.2106 | 134.3138 | 20.6318 | 0.9538 | 1.8118 |
| C5 | DINOv2-S | pointwise | 22056576 | 1.6343 | 150.2424 | 31.4855 | 0.9978 | 1.7518 |
| C6 | DINOv2-B | pointwise | 86580480 | 2.4972 | 200.5906 | 79.4489 | 0.9973 | 1.6694 |
| C7 | MedSigLIP-448 | pointwise | 428565440 | 3.7553 | 1484.5511 | 1370.4594 | 0.9995 | 1.5346 |
| C8 | DINOv2-S | statistics | 22056576 | 1.9659 | 152.8125 | 28.8735 | 0.9973 | 1.2604 |
| C9 | CT dose network | pointwise | 1175142 | 1.5901 | 128.0857 | 8.7048 | 0.7242 | 0.3083 |
| C10 | CT dose network | statistics | 1175142 | 2.9105 | 132.1236 | 11.5900 | 0.7121 | 0.7544 |

### Compute Budget

| Item | Value |
|---|---|
| Trained configurations | 11 |
| Seeds | 3 |
| Full U-Net runs | 33 |
| CT dose network pretraining, GPU hours (measured) | 0.111 |
| Screen (Step 1.4), GPU hours (measured) | 0.238 |
| Calibration (Step 2.1), GPU hours (measured) | 0.086 |
| GPU hours per U-Net run, C0 (measured in Step 1.2) | 0.155 |
| GPU hours per U-Net run, C7 MedSigLIP (measured in Step 2.3) | 1.445 |
| GPU hours of the seed 42 U-Net runs (11 of 11) | 3.259 |
| GPU hours used in Weeks 1 and 2 | 3.694 |
| Total GPU hours of the protocol (3 seeds, from the seed 42 runs) | 10.211 |
| GPU | NVIDIA A100-SXM4-80GB (79.3 GB) |
| Largest peak GPU memory of a run, GB | 17.0 |
| GPU hours available in Weeks 1 to 3 | fill in by hand |
