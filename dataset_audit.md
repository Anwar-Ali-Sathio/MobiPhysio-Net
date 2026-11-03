# Dataset audit - MobiPhysio (doi:10.7910/DVN/XSI0QN, version 3.0)

Landing page: https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/XSI0QN  
License: CC0 1.0  
Release time: 2026-02-11T08:46:02Z  

## Coverage (counts from saved manifests)

| Quantity | Count |
|---|---:|
| Files in Dataverse inventory | 3688 |
| Videos in inventory | 3686 |
| Inventory size (GB) | 158.71 |
| Restricted files | 0 |
| Duplicate-checksum groups (exact duplicate content) | 1 |
| Videos downloaded + MD5-verified | 8 |
| Videos decoded + cached (16 frames) | 8 |
| Download failures | [] |
| Decode failures | [] |
| Filenames parsed to valid labels | 3686 |
| Usable for Task A (labelled + decoded) | 916 |
| Participants (usable) | 56 |

Raw-data mode: stream-process: raw MP4 deleted after checksum verification + frame decoding. Dataset complete in cache: **False**.

## Exclusions

2770 videos excluded.

| video_id | exclusion_reason |
|---|---|
| E01_P59_AR_VFL_GF | preprocess: not_attempted |
| E01_P60_AF_VFL_GF | preprocess: not_attempted |
| E01_P60_AF_VHJ_GF | preprocess: not_attempted |
| E01_P60_AF_VLJ_GF | preprocess: not_attempted |
| E01_P60_AF_VLL_GF | preprocess: not_attempted |
| E01_P60_AF_VLR_GF | preprocess: not_attempted |
| E01_P60_AF_VML_GF | preprocess: not_attempted |
| E01_P60_AF_VO_GF | preprocess: not_attempted |
| E01_P60_AL_VFL_GF | preprocess: not_attempted |
| E01_P60_AR_VFL_GF | preprocess: not_attempted |
| E01_P61_AF_VFL_GF | preprocess: not_attempted |
| E01_P61_AF_VO_GF | preprocess: not_attempted |
| E01_P61_AL_VFL_GF | preprocess: not_attempted |
| E01_P62_AF_VFL_GF | preprocess: not_attempted |
| E01_P62_AF_VHJ_GF | preprocess: not_attempted |
| E01_P62_AF_VLJ_GF | preprocess: not_attempted |
| E01_P62_AF_VLL_GF | preprocess: not_attempted |
| E01_P62_AF_VLR_GF | preprocess: not_attempted |
| E01_P62_AF_VO_GF | preprocess: not_attempted |
| E01_P62_AL_VFL_GF | preprocess: not_attempted |
| E02_P01_AF_VFL_GM | preprocess: not_attempted |
| E02_P01_AF_VHJ_GM | preprocess: not_attempted |
| E02_P01_AF_VLJ_GM | preprocess: not_attempted |
| E02_P01_AF_VLL_GM | preprocess: not_attempted |
| E02_P01_AF_VML_GM | preprocess: not_attempted |
| E02_P01_AF_VO_GM | preprocess: not_attempted |
| E02_P01_AL_VFL_GM | preprocess: not_attempted |
| E02_P01_AL_VHJ_GM | preprocess: not_attempted |
| E02_P01_AL_VLJ_GM | preprocess: not_attempted |
| E02_P01_AL_VLL_GM | preprocess: not_attempted |
| E02_P01_AL_VML_GM | preprocess: not_attempted |
| E02_P01_AL_VO_GM | preprocess: not_attempted |
| E02_P01_AR_VFL_GM | preprocess: not_attempted |
| E02_P01_AR_VHJ_GM | preprocess: not_attempted |
| E02_P01_AR_VLL_GM | preprocess: not_attempted |
| E02_P01_AR_VML_GM | preprocess: not_attempted |
| E02_P01_AR_VO_GM | preprocess: not_attempted |
| E02_P02_AF_VHJ_GM | preprocess: not_attempted |
| E02_P02_AF_VLJ_GM | preprocess: not_attempted |
| E02_P02_AF_VML_GM | preprocess: not_attempted |
| E02_P02_AF_VO_GM | preprocess: not_attempted |
| E02_P02_AL_VFL_GM | preprocess: not_attempted |
| E02_P02_AL_VHJ_GM | preprocess: not_attempted |
| E02_P02_AL_VLL_GM | preprocess: not_attempted |
| E02_P02_AL_VML_GM | preprocess: not_attempted |
| E02_P02_AL_VO_GM | preprocess: not_attempted |
| E02_P02_AR_VFL_GM | preprocess: not_attempted |
| E02_P02_AR_VHJ_GM | preprocess: not_attempted |
| E02_P02_AR_VLL_GM | preprocess: not_attempted |
| E02_P02_AR_VML_GM | preprocess: not_attempted |

## Label / metadata distribution (usable videos)

| exercise | exercise_name | videos |
|---|---|---|
| E01 | Abduction | 538 |
| E02 | Adduction | 50 |
| E03 | Lateral Rotation | 61 |
| E04 | Medial Rotation | 45 |
| E05 | Circumduction | 59 |
| E06 | Wrist Extension | 58 |
| E07 | Hip Joint Flexion | 30 |
| E08 | Both Leg Flexion | 40 |
| E09 | Back Extension | 35 |

- **angle**: F=405, L=262, R=249
- **variation**: FL=205, HJ=118, LJ=117, LL=160, LR=18, ML=160, O=138
- **gender**: F=253, M=663

- Imbalance: largest/smallest class = 538/30 = 17.93.
- Participant coverage is uneven: participants contribute between 1 and 41 videos, and many participants performed only a subset of the exercises.

## Discrepancies vs the research brief

- E08 is named **'Both Leg Flexion'** in the official README (MobiPhysio.md v1.1), not 'Lumbar Flexion'. The official name is used everywhere.


## EAAQ annotations (Task B)

- Annotation blocks parsed: 3022 (named 3010, **unnamed 12**). Unnamed blocks are NOT linked to any video (no inference from row order).
- Duplicate video names in the CSV: 0 (excluded from linkage).
- Blocks with a judge count other than 3: 11.
- Stored average != mean of judges' 0-100 scores (>0.01): 0.
- Named blocks not matching any inventory video: 0.
- Videos with a linked score: 3010; without: 676.
- Score range in CSV: 0.00 - 100.00 (0-100, higher = better execution).
- Schema: 4 'Primary Factors' + 3 'Control Factors' items per judge (each 0-1, partial credit allowed), total mark 0-7 rescaled to 0-100; per-video target = mean over the three judges. Target level: per video. Type: continuous (bounded).

## Participant-independent split

Split file: `data/manifests/splits_pilot.csv` (sha256 `455f9786d4dd684a...`).

| Split | Videos | Participants |
|---|---:|---:|
| train | 280 | 35 |
| val | 38 | 6 |
| test | 82 | 10 |

- Participant overlap between any splits: **False**
- Exact-duplicate (MD5) content across splits: 0
- Test participants: P05, P07, P10, P14, P19, P27, P34, P40, P45, P55

- Near-duplicate audit (cosine of mean frozen-ViT embedding, threshold 0.98): 0 cross-split pairs above threshold; max cross-split cosine test-train=0.9762, test-val=0.9524, train-val=0.9731.
