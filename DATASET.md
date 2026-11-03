# MobiPhysio dataset — integration notes

* DOI `doi:10.7910/DVN/XSI0QN` → https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/XSI0QN
  (resolved programmatically by `mobiphysio inventory`; result in `data/manifests/dataset_metadata.json`).
* Version used: **3.0** (RELEASED), license **CC0 1.0**, 3,688 files: 3,686 MP4 videos + `MobiPhysio.md`
  (README v1.1) + `Physiotherapist Exercise Marking .csv`. No restricted files. Total ≈ 158.7 GB.
* Every file carries an MD5 in the Dataverse metadata; every downloaded file is verified against it.

## Access quirk

Harvard Dataverse answers **HTTP 403** to the default `python-requests` User-Agent (observed
2026-10-27); the client therefore always sends `mobiphysio-research/0.1 (+https://doi.org/10.7910/DVN/XSI0QN)`.

## Storage strategy

The full dataset is ~5× the free disk of the execution host (31 GB). `mobiphysio preprocess` therefore
**streams**: download (HTTP Range resume, 5 retries with exponential back-off) → MD5 check → decode 16
frames → cache → delete the MP4. The decoded cache (16×224×224×3 uint8 per video) is ~9 GB. Use
`mobiphysio download` (or `keep_raw_videos: true`) to retain raw files instead. `download_report.json`
never labels a partial run as complete (`complete` is true only if *every* inventory video is decoded).

## Filename convention (README Table 2)

`E{xx}_P{yy}_A{F|L|R}_V{FL|ML|LL|LJ|HJ|O|LR}_G{M|F}.mp4`, parsed by the configurable regex in
`configs/dataset.yaml`. An unknown exercise code or missing participant is an **error** (video excluded,
reason recorded); an undocumented angle/variation code is a **warning** only (exercise label still valid).
One file uses variation `VP`, which is not in the README codebook.

## Classes

E01 Abduction · E02 Adduction · E03 Lateral Rotation · E04 Medial Rotation · E05 Circumduction ·
E06 Wrist Extension · E07 Hip Joint Flexion · **E08 Both Leg Flexion** · E09 Back Extension.
(The research brief called E08 "Lumbar Flexion"; the official README name is used.)

## EAAQ annotations

`Physiotherapist Exercise Marking .csv`: two header rows, then one block per video with one row per
judge (three physiotherapists). Items: 4 "Primary Factors" + 3 "Control Factors" (0–1, partial credit),
total 0–7 → 0–100; the block's first row also stores the judges' average (0–100, higher = better).

Linkage is **by explicit video name only**. Blocks without a name are never assigned (no inference from
row order). Observed: 3,022 blocks, 3,010 uniquely named and all matching inventory videos, 1 unnamed
block with real scores (CSV row 27), 11 empty spreadsheet-residue blocks at the end of the file, no
duplicate names, no average/recomputation mismatches. 676 videos have no linked score.

## Participants

58 participant IDs (P01–P62 minus 25, 44, 51, 54, which the README says were removed during curation).
Contributions are very uneven (7–162 videos); many participants recorded only a subset of exercises.
The README states 39 male / 19 female. No clinical-population information is given in the README;
see the associated Data in Brief article for the ethics statement and recruitment details.

## Duplicate content found in the published files

* `E01_P02_AR_VFL_GM.mp4` and `E01_P06_AR_VLL_GM.mp4` are **byte-identical** (MD5
  `8c3d32e316bbd5bd039d3f9d57794a18`, same size, identical decoded frames) but are labelled as different
  participants (P02 / P06) and variations (VFL / VLL). At least one label is wrong and we cannot tell which,
  so both are excluded from Task A (`exclusion_reason` in `video_manifest.csv`). 3,684 videos remain usable.
* The cross-split near-duplicate audit (embedding cosine ≥ 0.98, then mean |pixel diff| < 2) nominated
  candidates that, on visual review, are *different people filmed in the same room with the same camera
  and background* (`data/manifests/near_duplicate_manual_review.json`). Many recordings share this scene,
  so the background carries no label information but makes clip-level similarity high.
