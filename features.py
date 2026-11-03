"""Backbone feature extraction from the frame cache (Algorithm 2 for frozen ViT).

Outputs in ``<data_dir>/features``:
* ``vit_s16.npz``                CLS features after final norm, (N, T, 384) float16
* ``vit_s16_lastblock_in.f16``   tokens entering the final ViT block, memmap
                                 (N, T, 197, 384) float16 (for last-block fine-tuning)
* ``resnet50.npz``               ImageNet ResNet50 pooled features, (N, T, 2048) float16
* ``frames112.u8``               112x112 RGB frames for from-scratch CNNs, memmap
* ``svm_descriptors.npz``        handcrafted video descriptors, (N, D) float32
"""
from __future__ import annotations

import json
import logging
import time

import cv2
import numpy as np
import torch

from .cache import FeatureStore, FrameCache

log = logging.getLogger(__name__)

VIT_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"


def load_vit(pretrained: bool = True):
    import timm
    m = timm.create_model(VIT_NAME, pretrained=pretrained, num_classes=0).eval()
    return m


def vit_preprocess(frames_u8: np.ndarray, mean, std) -> torch.Tensor:
    x = torch.from_numpy(frames_u8).float().div_(255.0).permute(0, 3, 1, 2)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def vit_forward_split(vit, x: torch.Tensor, run_last: bool = True):
    """Return (tokens entering the last block, CLS after last block + norm | None)."""
    t = vit.patch_embed(x)
    t = vit._pos_embed(t)
    t = vit.patch_drop(t)
    t = vit.norm_pre(t)
    for blk in vit.blocks[:-1]:
        t = blk(t)
    pre = t
    if not run_last:
        return pre, None
    out = vit.norm(vit.blocks[-1](t))
    return pre, out[:, 0]


def _sidecar(store: FeatureStore, name: str, info: dict) -> None:
    (store.root / f"{name}.json").write_text(json.dumps(info, indent=2))


def extract_vit(video_ids: list[str], cache: FrameCache, store: FeatureStore, t: int,
                batch_videos: int = 2) -> dict:
    vit = load_vit(True)
    cfg = vit.pretrained_cfg
    n = len(video_ids)
    n_tok = vit.patch_embed.num_patches + 1
    d = vit.num_features
    tok_path = store.root / "vit_s16_lastblock_in.f16"
    tok = np.lib.format.open_memmap(str(tok_path) + ".npy", mode="w+", dtype=np.float16,
                                    shape=(n, t, n_tok, d))
    cls = np.zeros((n, t, d), np.float16)
    masks = np.zeros((n, t), bool)
    t0 = time.time()
    for s in range(0, n, batch_videos):
        ids = video_ids[s:s + batch_videos]
        fr, mk = zip(*[cache.load(v) for v in ids])
        x = vit_preprocess(np.concatenate(fr), cfg["mean"], cfg["std"])
        pre, c = vit_forward_split(vit, x)
        pre = pre.view(len(ids), t, n_tok, d).numpy().astype(np.float16)
        c = c.view(len(ids), t, d).numpy()
        if not np.isfinite(c).all():
            raise FloatingPointError(f"non-finite ViT features for {ids}")
        tok[s:s + len(ids)] = pre
        cls[s:s + len(ids)] = c.astype(np.float16)
        masks[s:s + len(ids)] = np.stack(mk)
        if (s // batch_videos) % 100 == 0:
            log.info("vit %d/%d %.0fs", s + len(ids), n, time.time() - t0)
    tok.flush()
    del tok
    store.save("vit_s16", video_ids, cls, masks)
    info = {"model": VIT_NAME, "source": "timm / Hugging Face Hub (augreg in21k -> in1k)",
            "pretrained": True, "embedding_dim": d, "tokens": n_tok, "mean": cfg["mean"],
            "std": cfg["std"], "input_size": cfg["input_size"], "seconds": time.time() - t0,
            "n_videos": n, "feature": "CLS token after final LayerNorm"}
    _sidecar(store, "vit_s16", info)
    return info


def extract_resnet50(video_ids: list[str], cache: FrameCache, store: FeatureStore, t: int,
                     batch_videos: int = 2) -> dict:
    from ..models.resnet50 import resnet50_backbone
    m = resnet50_backbone(True)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    n = len(video_ids)
    feats = np.zeros((n, t, 2048), np.float16)
    masks = np.zeros((n, t), bool)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch_videos):
            ids = video_ids[s:s + batch_videos]
            fr, mk = zip(*[cache.load(v) for v in ids])
            x = vit_preprocess(np.concatenate(fr), mean, std)
            f = m(x).view(len(ids), t, -1).numpy()
            if not np.isfinite(f).all():
                raise FloatingPointError(f"non-finite ResNet50 features for {ids}")
            feats[s:s + len(ids)] = f.astype(np.float16)
            masks[s:s + len(ids)] = np.stack(mk)
            if (s // batch_videos) % 100 == 0:
                log.info("resnet50 %d/%d %.0fs", s + len(ids), n, time.time() - t0)
    store.save("resnet50", video_ids, feats, masks)
    info = {"model": "torchvision resnet50 IMAGENET1K_V2", "pretrained": True,
            "embedding_dim": 2048, "mean": mean, "std": std, "seconds": time.time() - t0,
            "n_videos": n}
    _sidecar(store, "resnet50", info)
    return info


def extract_small_frames_and_svm(video_ids: list[str], cache: FrameCache, store: FeatureStore,
                                 t: int, small: int = 112) -> dict:
    from ..models.svm import video_descriptor
    n = len(video_ids)
    arr = np.lib.format.open_memmap(str(store.root / "frames112.u8.npy"), mode="w+",
                                    dtype=np.uint8, shape=(n, t, small, small, 3))
    masks = np.zeros((n, t), bool)
    desc = []
    t0 = time.time()
    for i, v in enumerate(video_ids):
        fr, mk = cache.load(v)
        arr[i] = np.stack([cv2.resize(f, (small, small), interpolation=cv2.INTER_AREA) for f in fr])
        masks[i] = mk
        desc.append(video_descriptor(fr, mk))
    arr.flush()
    del arr
    np.save(store.root / "frames112_mask.npy", masks)
    np.save(store.root / "frames112_ids.npy", np.asarray(video_ids))
    np.savez(store.path("svm_descriptors"), video_ids=np.asarray(video_ids),
             feats=np.stack(desc), mask=masks)
    info = {"frames112": [n, t, small, small, 3], "svm_descriptor_dim": int(desc[0].shape[0]),
            "seconds": time.time() - t0}
    _sidecar(store, "frames112_svm", info)
    return info
