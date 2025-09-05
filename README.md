# Datasets for Learned Inertial Odometry (IO) — Supplementary Data for “Extended Model‑Based Learned Inertial Odometry”

This folder contains curated and simplified datasets tailored for training and evaluating inertial odometry models. It serves as the supplementary data source for the paper “Extended Model‑Based Learned Inertial Odometry” by Sasanka Kuruppu Arachchige and Joni Kamarainen (IEEE conference, 2025 — tentative). Data are standardized to a lightweight, Blackbird‑style layout: per‑sequence CSVs for IMU, ground‑truth poses, and (when available) thrust. These are derived/simplified from their original sources for IO use only; please refer to and cite the original datasets when appropriate.

Status: active; contents align with the processing scripts in `RAW/`.

## What’s here
- Blackbird/ — pre-arranged sequences in Blackbird layout (train/eval/test by pattern: clover, egg, halfMoon, star, winter)
- DIDO/ — sequences produced from raw HDF5 (see `RAW/process_dido.py`)
- IIT-DRD/ — sequences produced from the TII Drone Racing dataset (see `RAW/process_drd.py`)
- EuRoC-Dataset/ — standard EuRoC MAV sequences with list files
- PegasusDataset/ — vehicle IMU dataset in CSV form (train/test folders)
- RAW/ — processing scripts and notes (`process_drd.py`, `process_dido.py`, `readme_process.md`)

All of the above are simplified for IO: only IMU, poses, and optional thrust signals are retained and normalized into common CSVs and folder structure for model training.

## Unified sequence layout (simplified for IO)
Each sequence folder follows this minimal format:
- groundTruthPoses.csv: `timestamp(ns), p_x, p_y, p_z, q_w, q_x, q_y, q_z`
- imu_data.csv: `timestamp(sec_from_start), accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z`
- thrust_data.csv (if available):
    - DRD (IIT-DRD): `timestamp(ns), motor0, motor1, motor2, motor3`
    - DIDO: `timestamp(sec_from_start), thrust_1, thrust_2, thrust_3, thrust_4`
- Optional: trajectory plot image (2D XY and optional oblique 3D)

Conventions
- Units: SI (m, s, rad, m/s^2)
- World frame: ENU; body: x‑forward, y‑left, z‑up (as provided by source, harmonized where noted by scripts)
- Timestamps: poses in nanoseconds; IMU timestamps are relative seconds from the first sample

## Processing from raw sources
Two helper scripts convert raw datasets into the above layout. See `RAW/readme_process.md` for details.

1) TII Drone Racing Dataset → IIT‑DRD
- Script: `RAW/process_drd.py`
- Input (example): `RAW/drone-racing-dataset/data/{autonomous|piloted}/flight-XX*/<flight>_500hz_freq_sync.csv`
- Output root (default): `IIT-DRD/`
- Splits: `train/`, `eval/`, `test/` via list files or random split
- Notes: rotation matrices are projected to SO(3) then converted to quaternions; trajectory plots optional

2) DIDO raw HDF5 → DIDO
- Script: `RAW/process_dido.py`
- Input (default): `RAW/DidoRaw/<sequence>/data.hdf5`
- Output root (default): `DIDO/`
- Output structure: `train|eval|test/<pattern>/yawForward/<sequence>/...`
- Notes: timestamp scale auto‑detected (s/μs/ns); thrust falls back to zeros if missing; trajectory plots optional

Example invocations (non‑destructive unless `--overwrite` is set) are documented in `RAW/readme_process.md`.

## Per‑dataset notes and original sources
- Blackbird
    - Format: already in the target layout under `Blackbird/` (train/eval/test across patterns)
    - Original dataset: The Blackbird UAV dataset (MIT)
        - https://blackbird-dataset.mit.edu/
- IIT‑DRD (converted from TII Drone Racing Dataset)
    - Produced by `process_drd.py` from 500 Hz synchronized CSVs
    - Original dataset source:
        - TII Racing — Drone Racing Dataset: https://github.com/tii-racing/drone-racing-dataset
- DIDO
    - Produced by `process_dido.py` from HDF5 logs in `RAW/DidoRaw/`
    - Original source: internal/author‑provided sequences; contact maintainers if you need access
- EuRoC‑Dataset
    - Standard EuRoC MAV sequences; list files provided under `EuRoC-Dataset/`
    - Original dataset: EuRoC MAV
        - https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
- PegasusDataset
    - Vehicle IMU dataset with `imu_data.csv` and `ground_truth.csv` in train/test splits
    - Original source: project‑specific logs; public link may not be available

Please cite the original datasets in addition to any project papers when you use the data.

## Quick IO loading example
```python
import numpy as np
imu = np.genfromtxt('.../imu_data.csv', delimiter=',', names=True)
gt  = np.genfromtxt('.../groundTruthPoses.csv', delimiter=',', names=True)
```

## License and attribution
- This folder provides simplified/derived copies strictly for inertial‑odometry research.
- Respect original dataset licenses; see the linked sources for terms.

## Reference (tentative)
If these datasets or scripts are useful, please cite:

- Sasanka Kuruppu Arachchige and Joni Kamarainen, “Extended Model‑Based Learned Inertial Odometry,” in IEEE (conference TBD), 2025. [tentative]

BibTeX (placeholder):
```bibtex
@inproceedings{kuruppuArachchige2025emblio,
    title     = {Extended Model-Based Learned Inertial Odometry},
    author    = {Kuruppu Arachchige, Sasanka and Kamarainen, Joni},
    booktitle = {Proceedings of the IEEE Conference (TBD)},
    year      = {2025},
    note      = {Tentative venue; details to be updated}
}
```

## Issues
Use GitHub Issues for questions, corrections, and split clarifications. See also `RAW/readme_process.md` for script options and edge‑case handling.