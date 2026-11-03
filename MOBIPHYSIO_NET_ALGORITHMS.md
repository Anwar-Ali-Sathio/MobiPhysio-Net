# MobiPhysio-Net — algorithms as implemented

This document maps Algorithms 1–8 of the design to code, and records every concrete choice that the
design left open. All values below come from `configs/` and were fixed **before** any test-set evaluation.

```
video ─► Alg.1 preprocessing ─► Alg.2 ViT-S/16 (shared per frame) ─► Z (T×384)
      ─► Alg.3 temporal Transformer ─► H (T×128), g = masked mean(H)
      ─► Alg.5/6 DTW(H, R_k) for the training-only reference bank (Alg.4) ─► q (9)   [no gradient]
      ─► u = [g ‖ standardise(q)] ─► MLP ─► softmax over 9 exercises
```

## Algorithm 1 — preprocessing (`preprocessing/video.py`, `preprocessing/frames.py`)

| Choice | Value |
|---|---|
| Frames | T = 16, uniformly spaced timestamps `linspace(0, (n−1)/fps, 16)` over the whole clip |
| Decoder | PyAV accurate seek (keyframe before target, decode forward, first frame with pts ≥ target − ½ frame); OpenCV sequential-grab fallback |
| Rotation | container display-matrix rotation applied (`np.rot90(frame, rotation // 90)`); verified pixel-identical to OpenCV auto-rotation on a sample |
| Geometry | letterbox: centre on a black square canvas (aspect preserved — videos are mostly 1080×1920 portrait), resize to 224×224 (`INTER_AREA`) |
| Colour | RGB uint8 cache; per-backbone normalisation applied at feature time (ViT: mean = std = 0.5) |
| Short videos | all frames used once, zero-padded at the end, `mask=False` on padding |
| Integrity | each MP4 is MD5-verified against the Dataverse checksum before decoding; raw file deleted afterwards (disk budget) |
| Exclusions | every failure is written to `status_T16_S224/<video>.json` with its reason |

The two decoders agree on 15/16 sampled indices on the reference video; the remaining index differs by one
frame (≈33 ms) because of slightly variable frame timing. All reported runs use the PyAV path
(`decoder` field per video in the status records).

## Algorithm 2 — ViT spatial encoder (`preprocessing/features.py`)

* Model: `timm` `vit_small_patch16_224.augreg_in21k_ft_in1k` (ImageNet-21k pre-training, ImageNet-1k
  fine-tuning; weights from the Hugging Face Hub via timm). 12 blocks, 384-d, 197 tokens.
* Frame feature `z_t` = CLS token after the final LayerNorm.
* **Why ViT-S and not ViT-B:** measured CPU throughput on the 4-core host was 23.8 img/s (ViT-S) vs
  8.1 img/s (ViT-B); 3,686×16 = 58,976 frames per pass makes ViT-B ≈2 h per pass.
* **Frozen** (primary configuration): features are computed once; this is mathematically identical to
  running the frozen network inside the model.
* **Last-block fine-tuning** (ablation 5): the tokens *entering* block 12 are cached (float16,
  N×16×197×384); `LastBlockViT` holds trainable copies of block 12 and the final norm, so gradients
  genuinely update ViT weights. Full 12-block fine-tuning was not feasible on CPU.

## Algorithm 3 — temporal encoder (`models/mobiphysio_net.py: TemporalEncoder`)

`Linear(384→128) + LayerNorm` → + fixed sinusoidal positions → 2-layer pre-norm Transformer encoder
(4 heads, FFN 256, dropout 0.2) with `src_key_padding_mask = ¬mask` → LayerNorm → padded positions zeroed.
Pooling: `g = Σ m_t h_t / max(1, Σ m_t)`. A unit test asserts that changing padded inputs changes neither
the logits nor the DTW distances.

## Algorithm 4 — training-only reference bank (`models/references.py`)

* Space for *selection*: frozen ViT CLS sequences, L2-normalised per frame (selection must not depend on
  the network being trained).
* Per class: up to 200 **training** videos (seeded random subset) → full pairwise DTW matrix (Alg. 5,
  same band) → **k-medoids with k = 3** (initialised with the class DTW medoid + farthest-point additions,
  then alternating assignment / medoid update). k = 3 was chosen a priori because every repetition was
  filmed from three camera angles.
* Saved to `experiments/<exp>/reference_bank.json` with video IDs, SHA-256 of the training-ID list and of
  the reference-ID list. `verify_reference_provenance` raises `LeakageError` if any reference is not a
  training video or is a validation/test video; it is re-checked at load time and by `verify-results`.
* The same 27 reference videos are used by every DTW variant (full, no-temporal, fine-tuned ViT).

## Algorithm 5 — constrained DTW (`models/dtw.py`)

* Local cost `c(i,j) = ‖h_i − r_j‖₂`; recursion over (i−1,j−1), (i−1,j), (i,j−1), `D(0,0)=0`, borders `+∞`.
* Deterministic tie-break: diagonal, then vertical, then horizontal. Path length `|π|` is carried through
  the recursion (`P(i,j) = P(pred)+1`) and the path is recoverable by back-pointers (NumPy reference).
* **Normalisation:** `d = D(T, L) / |π|` (length of the selected optimal path).
* **Constraint:** Sakoe-Chiba band on the length-normalised diagonal, `|i − j·T/L| ≤ w`,
  `w = max(4, |T−L|)` (the widening guarantees the end cell is reachable).
* Masks: padding is always a suffix; only the valid prefix participates (validated; a non-prefix mask is
  an error).
* Two implementations: `dtw_numpy` (reference, with path) and `dtw_batch` (all-pairs PyTorch, under
  `torch.no_grad`). Tests check them against each other, against brute-force path enumeration, for
  self-distance 0, unequal lengths, band behaviour, non-finite input and invalid masks.
* **DTW is not differentiable here.** It is a no-gradient feature branch; a test asserts that the
  distance tensor has no gradient while the temporal encoder still receives gradients from the loss.

## Algorithm 6 — fusion (`MobiPhysioNet.forward`)

1. Reference sequences are re-encoded with the **current** temporal encoder (eval mode, no gradient).
2. `q_k = min_{r ∈ R_k} d_DTW(H, r)` (aggregation rule fixed in advance: `min`).
3. **Leave-one-out:** when a training video is itself a reference, that pair is masked to `+∞`, so a
   training video can never see a zero distance to itself (otherwise the head could learn a shortcut that
   cannot exist at test time).
4. `DistanceScaler`: per-class mean/std standardisation **fitted on training videos only** (≤ 2,048,
   seeded subset) — refitted with the reference re-encoding after every training epoch, before validation.
   The scaler and reference encodings in force at the selected epoch are the ones restored and used for
   testing (they are model buffers, so they are saved in the checkpoint).
5. `u = [g ‖ q̃]` → `LayerNorm → Dropout → Linear(137→128) → GELU → Dropout → Linear(128→9)` → softmax.
6. Saved per test video: true label, prediction, 9 probabilities, 9 raw DTW class distances.

## Algorithm 7 — training (`training/trainer.py`, `training/runners.py: run_mobiphysio`)

AdamW (lr 5·10⁻⁴, weight decay 0.05), one-cycle schedule (10 % warm-up), batch 32, ≤ 40 epochs,
class-weighted cross-entropy (weights `N / (K·n_k)` from **training** counts), gradient clipping 5,
early stopping (patience 8). Checkpoint selection: **maximum validation macro F1** (first occurrence).
Seeds are set for Python/NumPy/PyTorch; deterministic algorithms are requested (warn-only).

## Algorithm 8 — evaluation (`evaluation/`)

Accuracy, balanced accuracy, macro P/R/F1, weighted F1, per-class P/R/F1, confusion matrix, one-vs-rest
macro ROC-AUC and macro average precision (only if all 9 classes occur in the test set and the
probabilities are valid), participant-cluster bootstrap 95 % CIs, paired bootstrap for ablation
differences, McNemar discordant counts, CPU latency (median over test videos, batch of one video, from
cached frames, backbone included) and parameter counts. Each test video is predicted exactly once by the
selected checkpoint. `verify-results` recomputes every metric from the prediction CSVs.

## Ablation variants (`configs/models/mobiphysio_net.yaml: ablations`)

| Variant | ViT | Temporal encoder | DTW branch |
|---|---|---|---|
| full (= frozen ViT) | frozen | yes | yes |
| no_dtw | frozen | yes | no (`u = g`) |
| dtw_no_temporal | frozen | no — DTW on L2-normalised ViT features, `g` = masked mean of Z | yes |
| finetuned_vit | last block + norm trainable | yes | yes (lr 2·10⁻⁴, ≤ 8 epochs, patience 3, batch 16, scaler fitted on 512 training videos — CPU budget fixed from the pilot epoch time, before any full-data test evaluation) |
