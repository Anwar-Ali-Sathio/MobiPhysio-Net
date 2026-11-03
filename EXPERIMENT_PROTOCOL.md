# Experiment protocol

## Tasks
* **Task A (primary):** 9-class exercise recognition from video.
* **Task B (secondary, exploratory):** regression of the per-video mean EAAQ score (0–100) of three
  physiotherapists, using only explicitly named annotation blocks. Task B is a feasibility probe, not a
  validated movement-quality assessor; recognition accuracy (Task A) says nothing about correctness.

## Split (`mobiphysio split`)
Participant-grouped 70/10/20 (README recommendation). Among 2,000 seeded random participant partitions,
the one minimising *max class-proportion deviation + max video-fraction deviation* is kept, subject to
every class appearing in every split. Only labels are used — no model. The assignment is saved
(`data/manifests/splits_full.csv`) with its SHA-256 and a leakage audit (participant overlap, cross-split
MD5 duplicates, cross-split near-duplicates by cosine ≥ 0.98 of mean frozen-ViT embeddings). The three
simultaneous camera angles of a repetition belong to one participant and therefore one split.

## Fitting rules
Everything learned is fitted on **train**: standard scaler (SVM), class weights, DTW reference bank,
DTW distance standardiser, network weights. **Validation** is used for SVM C/γ, checkpoint/epoch
selection (max macro F1) and early stopping. **Test** is touched once per run, after selection. The
primary MobiPhysio-Net configuration (frozen ViT, k = 3 references, `min` aggregation, band 4) and all
ablation definitions were fixed before any test evaluation.

## Models
| Model | Input | Trainable part |
|---|---|---|
| SVM | HOG (64×64, 8×8 cells, 9 bins; temporal mean+std) + 16×16 motion-energy statistics + motion profile | RBF SVC, class-balanced, C ∈ {0.1,1,10,100}, γ ∈ {0.1,0.3,1,3}/D |
| 2D-CNN | RGB 16×112×112, from scratch | compact 4-stage CNN (stride-2 stem, ≈0.18 GFLOP/frame) per frame, mean of frame logits |
| LSTM | frozen ResNet50 frame embeddings (2048) | Linear 256 → LSTM 256 → last valid state |
| BiLSTM | same | Linear 256 → BiLSTM 256 → [fwd ‖ bwd] final states |
| CNN-LSTM | RGB 16×112×112, from scratch | same CNN trunk + LSTM 128, end to end |
| ResNet50 | RGB 16×224×224, frozen ImageNet ResNet50 | linear head per frame, mean of frame logits |
| MobiPhysio-Net | RGB 16×224×224, frozen ViT-S/16 | temporal Transformer + DTW features + MLP |

The comparison is **not perfectly controlled**: inputs, pre-training and capacity differ (see table).
Backbones are frozen for CPU-budget reasons, which favours models built on strong pre-trained features.
Augmentation (shift-crop ±8 px + horizontal flip, same for all frames) is used only for the from-scratch
CNNs; they also see a sorted random subset of 8 of the 16 frames per training step (all 16 at evaluation)
for CPU budget. Pilot runs (`--experiment pilot`) use 400 videos and ≤ 3 epochs and live in `experiments/pilot/`.

## Uncertainty
Participant-level cluster bootstrap (2,000 replicates) for accuracy, balanced accuracy and macro F1;
paired bootstrap for ablation differences; retraining with seeds 1 and 2 (same split) for the cheaper
models to show training-seed variability. Split-to-split variability (a different participant split) is
NOT estimated — with ~12 test participants it is likely the dominant source of variance.
