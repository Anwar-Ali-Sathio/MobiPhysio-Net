# Limitations

## Clinical scope
* Task A answers *which* of nine AROM exercises is performed. It does **not** assess whether the movement
  is correct, safe or therapeutically adequate. No clinical validation, clinical utility or deployment
  readiness is claimed.
* Task B (EAAQ score regression) is exploratory: the score is an average of three raters over seven
  coarse items; agreement statistics and a condition-only baseline are reported alongside model errors
  precisely because a low error can arise from recording conditions or exercise identity rather than
  from movement quality.
* The README does not describe the participants as patients; it reports 58 participants (39 male,
  19 female). No age, diagnosis or impairment information is available in the downloaded files, so the
  population should be presumed to be predominantly healthy volunteers performing guided exercises
  (see the Data in Brief article for the authoritative description).

## Statistical
* Single-seed results are unstable at this scale: across training seeds the same configuration moves
  by up to ±0.05 macro F1, and removing two videos (v1 → v2) changed the order of the top two models.
  Ablation conclusions are therefore drawn from the seed-paired table, not from one seed's bootstrap CI
  (which captures only test-sample uncertainty).
* CPU latency was measured once per run on a shared 4-core VM; differences of tens of milliseconds between
  MobiPhysio-Net variants are timing noise.
* One dataset, one participant-independent split, no external validation set.
* ~12 test participants: bootstrap CIs are wide and approximate; split-to-split variance is not
  estimated. Seed reruns cover only training stochasticity.
* Participant contributions are very unbalanced (7–162 videos) and several exercises (E02, E04, E06,
  E07) come from a smaller subset of participants, so per-class results rest on few people.
* Class imbalance (largest/smallest ≈ 2.2) is handled by class-weighted losses / balanced SVC;
  macro F1 and balanced accuracy are reported alongside accuracy.

## Engineering compromises (CPU-only host: 4 cores, 15 GB RAM, no GPU, 31 GB disk)
* ViT-S/16 instead of ViT-B; ViT frozen in the primary model; only the **last** ViT block is fine-tuned in
  the fine-tuning ablation (with a reduced epoch budget).
* ResNet50 is a frozen ImageNet backbone with a trained linear head (linear probe), not a fine-tuned
  network; the LSTM/BiLSTM baselines use the same frozen ResNet50 embeddings.
* 2D-CNN and CNN-LSTM are trained from scratch at 112×112 because full-resolution training on CPU was
  too slow; they are therefore disadvantaged relative to the pre-trained-feature models.
* The SVM uses handcrafted descriptors; HOG was re-implemented in NumPy because OpenCV 5 no longer ships
  `cv2.HOGDescriptor`.
* No pose features were extracted (no pose estimator was installed/validated in this environment); all
  models use RGB-derived inputs.
* 16 uniformly sampled frames per ~30 s clip (≈ one frame every 2 s) under-sample fast movements.
* Raw videos were deleted after decoding; reprocessing requires re-downloading (~158.7 GB).

## Comparison validity
Model families differ in input resolution, pre-training and capacity, so the comparison ranks these
specific configurations, not architecture families. Hyper-parameters of the baselines were not
extensively tuned (only SVM C/γ via validation; networks use fixed, documented settings).

## Data
* 12 unnamed annotation blocks are not linked (1 real, 11 empty residue rows); 676 videos lack scores.
* One video uses the undocumented variation code `VP`.
* Two published files are byte-identical but carry different participant/variation labels; both are
  excluded. A first full run (v1, archived in `experiments/full_v1_superseded/`) had one copy in train and
  the other in validation; the reported v2 run fixes this (identical test set and hyper-parameters).
* Many participants were filmed in the same room with a fixed camera; models may exploit scene-specific
  cues, and generalisation to other environments is untested.
* Data are CC0, but they show identifiable people; this repository redistributes no frames or videos.
