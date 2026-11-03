# Reproducibility audit

Experiment: `pilot`  
Split assignment sha256: `455f9786d4dd684a47e796f3910e7ce4e051afd74aa23a50fd079cf714920914`

| Model key | Run ID | Seed | Best epoch | Val macro F1 | Started (UTC) | Ended (UTC) | Exit | Python | torch |
|---|---|---:|---:|---:|---|---|---|---|---|
| lstm | `lstm-20261027T144821Z-ad0464` | 42 | 3 | 0.276 | 2026-10-27T14:48:21Z | 2026-10-27T14:49:43Z | completed | 3.13.17 | 2.14.1+cpu |
| bilstm | `bilstm-20261027T144943Z-fcf931` | 42 | 3 | 0.224 | 2026-10-27T14:49:43Z | 2026-10-27T14:51:40Z | completed | 3.13.17 | 2.14.1+cpu |
| resnet50 | `resnet50-20261027T145140Z-30ef46` | 42 | 2 | 0.342 | 2026-10-27T14:51:40Z | 2026-10-27T14:52:18Z | completed | 3.13.17 | 2.14.1+cpu |
| svm | `svm-20261027T145903Z-696e2c` | 42 | None | n/a | 2026-10-27T14:59:03Z | 2026-10-27T14:59:13Z | completed | 3.13.17 | 2.14.1+cpu |
| mobiphysio_net | `mobiphysio_net-20261027T145917Z-a3f86f` | 42 | 1 | 0.286 | 2026-10-27T14:59:17Z | 2026-10-27T15:01:31Z | completed | 3.13.17 | 2.14.1+cpu |
| cnn2d | `cnn2d-20261027T151355Z-61395f` | 42 | 3 | 0.114 | 2026-10-27T15:13:55Z | 2026-10-27T15:16:57Z | completed | 3.13.17 | 2.14.1+cpu |
| cnn_lstm | `cnn_lstm-20261027T151701Z-5f254c` | 42 | 2 | 0.121 | 2026-10-27T15:17:01Z | 2026-10-27T15:20:53Z | completed | 3.13.17 | 2.14.1+cpu |
| mobiphysio_net__no_dtw | `mobiphysio_net__no_dtw-20261027T152057Z-4bf40b` | 42 | 3 | 0.379 | 2026-10-27T15:20:57Z | 2026-10-27T15:22:40Z | completed | 3.13.17 | 2.14.1+cpu |
| mobiphysio_net__dtw_no_temporal | `mobiphysio_net__dtw_no_temporal-20261027T152240Z-999cf7` | 42 | 1 | 0.258 | 2026-10-27T15:22:40Z | 2026-10-27T15:23:41Z | completed | 3.13.17 | 2.14.1+cpu |
| mobiphysio_net__finetuned_vit | `mobiphysio_net__finetuned_vit-20261027T152341Z-e7fd06` | 42 | 3 | 0.285 | 2026-10-27T15:23:41Z | 2026-10-27T15:32:16Z | completed | 3.13.17 | 2.14.1+cpu |

## Environment

- Platform: Linux-6.18.44-fc-v165-x86_64-with-glibc2.39
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz x4, RAM 16.9 GB, GPU: none (CPU only)
- Packages: torch=2.14.1+cpu, torchvision=0.29.1+cpu, timm=1.0.30, numpy=2.5.3, pandas=3.0.5, scikit-learn=1.9.1, scipy=1.18.1, opencv-python-headless=5.0.0.93, av=19.0.2, matplotlib=3.11.2, PyYAML=6.0.3, requests=2.34.2, pytest=9.1.1, huggingface_hub=2.3.0, safetensors=0.8.0

## Determinism

- Seeds set for Python, NumPy, PyTorch; `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Multi-threaded CPU reductions can differ in final bits across machines; SVC Platt scaling uses internal (seeded) CV.
- Each run writes to its own `experiments/<experiment>/<model>/<run_id>/` directory; nothing is overwritten.