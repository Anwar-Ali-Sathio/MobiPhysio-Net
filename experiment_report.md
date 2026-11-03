# MobiPhysio-Net experiment report - PILOT (reduced data/epochs; not a benchmark)

All numbers below are read from saved prediction/metric files; see `reproducibility_audit.md` and `verification.json` for the recomputation check.

## Task A - 9-class exercise recognition (participant-independent test set)

| Model | Accuracy | Balanced acc. | Macro P | Macro R | Macro F1 | Macro F1 95% CI | Weighted F1 | ROC-AUC (OvR) | Test videos | Latency ms (median) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM | 0.610 | 0.189 | 0.208 | 0.189 | 0.159 | [0.087, 0.268] | 0.478 | 0.701 | 82 | 24.243 | COMPLETED (PILOT) |
| 2D-CNN | 0.171 | 0.071 | 0.116 | 0.071 | 0.070 | [0.038, 0.101] | 0.218 | 0.475 | 82 | 401.295 | COMPLETED (PILOT) |
| LSTM | 0.171 | 0.231 | 0.148 | 0.231 | 0.131 | [0.021, 0.197] | 0.188 | 0.747 | 82 | 4116.954 | COMPLETED (PILOT) |
| BiLSTM | 0.232 | 0.248 | 0.183 | 0.248 | 0.153 | [0.029, 0.236] | 0.262 | 0.743 | 82 | 4007.678 | COMPLETED (PILOT) |
| CNN-LSTM | 0.171 | 0.109 | 0.056 | 0.109 | 0.046 | [0.024, 0.067] | 0.182 | 0.439 | 82 | 489.293 | COMPLETED (PILOT) |
| ResNet50 | 0.366 | 0.284 | 0.183 | 0.284 | 0.196 | [0.114, 0.271] | 0.391 | 0.745 | 82 | 3831.842 | COMPLETED (PILOT) |
| MobiPhysio-Net | 0.378 | 0.176 | 0.127 | 0.176 | 0.142 | [0.075, 0.183] | 0.378 | 0.698 | 82 | 3201.511 | COMPLETED (PILOT) |

CI: participant-level cluster bootstrap (test participants resampled with replacement). Latency: median CPU time to classify one video from cached 16x224x224 frames (includes the backbone; excludes MP4 decoding).

### Input modality per model (comparison is NOT perfectly controlled)

| Model | Input / backbone | Trainable params | Total params (incl. frozen) |
|---|---|---|---|
| SVM | handcrafted HOG + motion-energy descriptor (16 frames) | 1209600 | 1209600 |
| 2D-CNN | RGB 16x112x112, from scratch | 457001 | 457001 |
| LSTM | frozen ImageNet ResNet50 frame embeddings | 1053193 | 24561225 |
| BiLSTM | frozen ImageNet ResNet50 frame embeddings | 1581833 | 25089865 |
| CNN-LSTM | RGB 16x112x112, from scratch | 653481 | 653481 |
| ResNet50 | RGB 16x224x224, frozen ImageNet ResNet50 + linear head | 18441 | 23526473 |
| MobiPhysio-Net | RGB 16x224x224, frozen ViT-S/16 + temporal Transformer + DTW | 333851 | 21999515 |

## Ablations (same split, same test videos)

| Variant | Macro F1 | Accuracy | Balanced acc. | Full - variant (macro F1) | diff CI low | diff CI high | Latency ms | Trainable params | Status |
|---|---|---|---|---|---|---|---|---|---|
| Full MobiPhysio-Net (frozen ViT) | 0.142 | 0.378 | 0.176 | n/a | n/a | n/a | 3201.511 | 333851 | COMPLETED (PILOT) |
| ViT + temporal encoder, no DTW | 0.160 | 0.195 | 0.217 | -0.018 | -0.092 | 0.112 | 2989.736 | 332681 | COMPLETED (PILOT) |
| ViT + DTW features, no temporal encoder | 0.160 | 0.305 | 0.231 | -0.018 | -0.131 | 0.069 | 2750.519 | 52379 | COMPLETED (PILOT) |
| Full model, frozen ViT (same run as row 1) | 0.142 | 0.378 | 0.176 | n/a | n/a | n/a | 3201.511 | 333851 | COMPLETED (PILOT) |
| Full model, last-block fine-tuned ViT | 0.205 | 0.561 | 0.234 | -0.063 | -0.103 | 0.046 | 3262.615 | 333851 | COMPLETED (PILOT) |

Difference CIs: paired participant-level bootstrap on identical test videos.

**Central ablation (DTW contribution):** full - no-DTW macro F1 = -0.0178 (95% CI [-0.0918, +0.1120]); the interval includes zero. Latency cost of the DTW branch: +211.8 ms per video. McNemar discordant counts: full right/no-DTW wrong = 24, full wrong/no-DTW right = 9.

## Training-seed variability (same split)

| model | n_seeds | seeds | macro_f1_mean | macro_f1_sd | per_seed_macro_f1 |
|---|---|---|---|---|---|
| LSTM | 1 | [42] | 0.131 | n/a | [0.1311] |
| BiLSTM | 1 | [42] | 0.153 | n/a | [0.1531] |
| ResNet50 | 1 | [42] | 0.196 | n/a | [0.1956] |
| SVM | 1 | [42] | 0.159 | n/a | [0.1594] |
| MobiPhysio-Net | 1 | [42] | 0.142 | n/a | [0.142] |
| 2D-CNN | 1 | [42] | 0.070 | n/a | [0.0697] |
| CNN-LSTM | 1 | [42] | 0.046 | n/a | [0.0463] |
| ViT + temporal (no DTW) | 1 | [42] | 0.160 | n/a | [0.1599] |
| ViT + DTW (no temporal encoder) | 1 | [42] | 0.160 | n/a | [0.1595] |
| Full, last-block fine-tuned ViT | 1 | [42] | 0.205 | n/a | [0.2053] |

Differences between models smaller than this seed spread should not be interpreted.

## MobiPhysio-Net by camera angle and recording variation (README-recommended breakdown)

| factor | level | n | accuracy | macro_f1 |
|---|---|---|---|---|
| angle | F | 34 | 0.441 | 0.093 |
| angle | L | 24 | 0.250 | 0.080 |
| angle | R | 24 | 0.417 | 0.151 |
| variation | FL | 21 | 0.429 | 0.127 |
| variation | HJ | 8 | 0.375 | 0.130 |
| variation | LJ | 13 | 0.462 | 0.083 |
| variation | LL | 10 | 0.300 | 0.061 |
| variation | LR | 2 | 1.000 | 0.111 |
| variation | ML | 14 | 0.429 | 0.181 |
| variation | O | 14 | 0.143 | 0.034 |

## Task B - EAAQ movement-quality score (exploratory)

Target: per-video mean of 3 physiotherapists' EAAQ scores (0-100), explicit-name linkage only. Linked usable videos: 311 (train 229, val 21, test 61). Inter-judge ICC(2,1) = 0.869; mean per-video judge SD = 2.92 points. Score mean 81.7 (SD 12.8).

| model | mae | rmse | r2 | spearman |
|---|---|---|---|---|
| train_mean | 9.802 | 11.953 | -0.100 | n/a |
| exercise_mean | 8.696 | 11.408 | -0.002 | 0.418 |
| exercise_variation_mean | 10.598 | 13.475 | -0.398 | 0.038 |
| ridge_vit | 9.799 | 12.180 | -0.142 | -0.095 |
| temporal_vit | 10.682 | 12.741 | -0.250 | -0.067 |

`exercise_variation_mean` uses only the exercise code and the recording-variation code. If it is competitive with the video models, the scores are largely explained by recording condition / exercise identity, and a low error does not demonstrate movement-quality assessment.

## Missing / incomplete

- All registered model runs completed.

## Figures

![model_comparison](../figures/model_comparison.png)
![confusion_matrices](../figures/confusion_matrices.png)
![per_class_f1](../figures/per_class_f1.png)
![training_curves](../figures/training_curves.png)
![dtw_ablation](../figures/dtw_ablation.png)

## Interpretation limits

- Recognising WHICH exercise is performed does not establish whether it is performed correctly; no clinical validity or utility is claimed.
- Single dataset, single participant split, no external validation; ~12 test participants make CIs wide.
- Backbones (ViT-S/16, ResNet50) are frozen ImageNet models (CPU budget); from-scratch CNNs see 112 px frames. Rankings reflect these specific, documented configurations - not the architecture families in general.
- See docs/LIMITATIONS.md.