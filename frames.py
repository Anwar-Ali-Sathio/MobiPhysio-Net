"""Frame sampling, letterboxing and padding masks (Algorithm 1, steps c-h)."""
from __future__ import annotations

import cv2
import numpy as np


def uniform_indices(n_frames: int, t: int) -> np.ndarray:
    """``t`` temporally ordered, uniformly spaced indices in ``[0, n_frames)``.

    If the video has fewer than ``t`` frames every frame is used once (no
    duplication); the caller pads and masks the remainder.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=int)
    if n_frames < t:
        return np.arange(n_frames)
    return np.linspace(0, n_frames - 1, t).round().astype(int)


def letterbox(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    """Pad to a square (black borders, centred) and resize; preserves aspect ratio."""
    h, w = frame_rgb.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=frame_rgb.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = frame_rgb
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def pad_sequence(frames: list[np.ndarray], t: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack frames to ``(t, size, size, 3)`` uint8 and a ``(t,)`` validity mask."""
    out = np.zeros((t, size, size, 3), dtype=np.uint8)
    mask = np.zeros(t, dtype=bool)
    for i, f in enumerate(frames[:t]):
        out[i] = f
        mask[i] = True
    return out, mask


def read_sampled_frames(path: str, t: int, size: int) -> dict:
    """Primary decoder: PyAV accurate seeking; falls back to OpenCV sequential reading."""
    try:
        r = read_sampled_frames_pyav(path, t, size)
        if r["ok"]:
            return r
        err = r["error"]
    except Exception as exc:  # noqa: BLE001 - any decoder failure triggers the fallback
        err = f"{type(exc).__name__}: {exc}"
    r = read_sampled_frames_cv2(path, t, size)
    r["pyav_error"] = err
    return r


def read_sampled_frames_pyav(path: str, t: int, size: int) -> dict:
    """Seek to the keyframe preceding each target timestamp and decode forward.

    Targets are ``t`` uniformly spaced timestamps from the first to the last frame,
    ``linspace(0, (n-1)/fps, t)``; for each, the first frame whose presentation time
    is >= target - half a frame is taken (frame-exact for constant frame rate).
    The container rotation (display matrix) is applied, matching OpenCV's
    auto-rotation (verified pixel-identical on a sample video).
    """
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate or s.guessed_rate or 0)
        tb = float(s.time_base)
        start = s.start_time or 0
        n = int(s.frames or 0)
        if n <= 0 and s.duration and fps:
            n = int(round(float(s.duration * s.time_base) * fps))
        if n < t or fps <= 0:
            return {"ok": False, "error": f"short or unknown-length stream (n={n}, fps={fps})"}
        targets = np.linspace(0, (n - 1) / fps, t)
        frames, times, shape, rot = [], [], None, 0
        for tt in targets:
            pts_target = start + int(round(tt / tb))
            c.seek(pts_target, stream=s, backward=True, any_frame=False)
            chosen = None
            for fr in c.decode(s):
                if fr.pts is None:
                    continue
                if (fr.pts - start) * tb >= tt - 0.5 / fps:
                    chosen = fr
                    break
            if chosen is None:
                return {"ok": False, "error": f"no frame at t={tt:.3f}s"}
            rot = int(getattr(chosen, "rotation", 0) or 0)
            img = chosen.to_ndarray(format="rgb24")
            if rot % 360:
                img = np.ascontiguousarray(np.rot90(img, rot // 90))
            shape = img.shape
            frames.append(letterbox(img, size))
            times.append(round(float((chosen.pts - start) * tb), 4))
        arr, mask = pad_sequence(frames, t, size)
        idx = [int(round(x * fps)) for x in times]
        h, w = shape[:2]
    return {"ok": True, "frames": arr, "mask": mask, "indices": idx, "timestamps": times,
            "reported_frames": n, "decoded_frames": n, "fps": fps, "width": w, "height": h,
            "rotation": rot, "decoded_shape": list(shape), "n_valid": int(mask.sum()),
            "padded": bool((~mask).any()), "decoder": "pyav_seek"}


def read_sampled_frames_cv2(path: str, t: int, size: int) -> dict:
    """Fallback: decode ``t`` uniformly sampled RGB frames by sequential grabbing.

    Sequential ``grab`` is used instead of seeking because seeking in HEVC streams
    is not frame-accurate. If the container's frame count is wrong (stream ends
    early) the indices are re-planned on the actual count and the file re-read.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "error": "unreadable (cv2.VideoCapture failed)"}
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    def _pass(n_plan: int):
        idx = uniform_indices(n_plan, t)
        wanted = set(idx.tolist())
        cap = cv2.VideoCapture(path)
        frames, k, last = [], 0, None
        while True:
            ok = cap.grab()
            if not ok:
                break
            if k in wanted:
                ok2, bgr = cap.retrieve()
                if ok2 and bgr is not None:
                    frames.append(letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size))
                    last = bgr.shape
            k += 1
        cap.release()
        return idx, frames, k, last

    idx, frames, actual, shape = _pass(max(reported, 1))
    if len(frames) < min(t, actual) and actual > 0:
        idx, frames, actual, shape = _pass(actual)
    if not frames:
        return {"ok": False, "error": f"no decodable frames (reported={reported})"}
    arr, mask = pad_sequence(frames, t, size)
    return {"ok": True, "frames": arr, "mask": mask, "indices": idx.tolist(),
            "reported_frames": reported, "decoded_frames": actual, "fps": fps,
            "width": width, "height": height,
            "decoded_shape": list(shape) if shape is not None else None,
            "n_valid": int(mask.sum()), "padded": bool((~mask).any()),
            "decoder": "cv2_sequential"}
