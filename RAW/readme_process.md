# Dataset Conversion Scripts: process_drd.py & process_dido.py

Unified interface for converting two raw datasets into a Blackbird-style
layout. Both scripts now share similar CLI options: split control via
list files or random ratios, optional plotting, overwrite behavior, and a
progress bar.

## 1. TII Drone Racing Dataset – `process_drd.py`

Input structure (example):
```
RAW/drone-racing-dataset/data/
  autonomous/flight-XXa-.../flight-XXa-..._500hz_freq_sync.csv
  piloted/flight-YYp-.../flight-YYp-..._500hz_freq_sync.csv
```

Source:
- TII Racing — Drone Racing Dataset: https://github.com/tii-racing/drone-racing-dataset
  - Tip: clone or symlink the repository under `RAW/drone-racing-dataset` so the defaults work.

Generated output (example):
```
IIT-DRD/
  train/flight-01a-ellipse/groundTruthPoses.csv
  train/flight-01a-ellipse/imu_data.csv
  train/flight-01a-ellipse/thrust_data.csv
  eval/...
  test/...
```

File semantics:
- groundTruthPoses.csv: `timestamp(ns), p_x, p_y, p_z, q_w, q_x, q_y, q_z`
- imu_data.csv: `timestamp(sec-from-first), accel_x..z, gyro_x..z`
- thrust_data.csv: `timestamp(ns), motor0..motor3`
- Optional plot (`flight_trajectory.png` by default): XY + (optional) 3D.

Quaternion derivation: rotation matrix `drone_rot[0..8]` projected to SO(3)
with SVD, then converted to (w,x,y,z). Pose and thrust timestamps use
nanoseconds; IMU timestamps are relative seconds starting at 0.

## 2. DIDO Dataset – `process_dido.py`

Input structure (default data-root):
```
RAW/DidoRaw/<sequence_dir>/data.hdf5
```

HDF5 datasets expected:
- ts: time (seconds, microseconds, or nanoseconds – auto detected)
- acc: (N,3) linear acceleration m/s^2
- gyr: (N,3) angular velocity rad/s
- gt_p: (N,3) position (m)
- gt_q: (N,4) quaternion (w,x,y,z)
- meas_rpm: (N,>=5) first column time-like (ignored), next 4 → thrust_1..4

Generated output (example):
```
DIDO/
  train/<pattern>/yawForward/<sequence>/groundTruthPoses.csv
  train/<pattern>/yawForward/<sequence>/imu_data.csv
  train/<pattern>/yawForward/<sequence>/thrust_data.csv
  eval/...
  test/...
```
`<pattern>` extracted from sequence name prefix up to the token 'a'.

File semantics (differences vs DRD):
- groundTruthPoses.csv identical column headers.
- imu_data.csv timestamps: default absolute nanoseconds (use --imu-timestamp-mode relative_sec for relative seconds).
- thrust_data.csv timestamps: default absolute nanoseconds (use --thrust-timestamp-mode relative_sec for relative seconds) + thrust_1..4.

## Split Handling (Both Scripts)
Priority order:
1. Explicit list files via `--train-list`, `--eval-list`, `--test-list`.
2. Auto-detected list files in the provided `--data-root` (DRD: train.txt, eval.txt, test.txt; DIDO: train.txt, val.txt or eval.txt, test.txt).
3. Random split using `--split-ratios` (must sum to 1.0) and `--seed`.

Notes:
- For DIDO, a `val.txt` or `eval.txt` is treated as the evaluation split (`eval` output folder).
- Missing names in list files are silently skipped.

## Common CLI Options
```
--data-root <path>       Raw data root
--out-dir <path>         Output dataset root
--train-list <file>      Explicit train list (one name per line)
--eval-list <file>       Explicit eval/val list
--test-list <file>       Explicit test list
--split-ratios a b c     Random ratios if no lists (default 0.7 0.15 0.15)
--seed <int>             RNG seed for random split (default varies: drd=0, dido=42)
--overwrite              Recreate existing flight/sequence outputs
--plots                  Enable trajectory plotting
--no-plot-3d             Disable the 3D subplot
--plot-name <name>       Plot filename (default differs: drd flight_trajectory.png, dido trajectory.png)
--plot-dpi <int>         Plot DPI (default 120)
--no-progress            Disable tqdm progress bar
(DRD only) --imu-timestamp-mode {relative_sec,ns}  IMU timestamps as absolute nanoseconds (default) or relative seconds from start
(DIDO only) --imu-timestamp-mode {relative_sec,ns}  Same meaning for DIDO sequences
(DIDO only) --thrust-timestamp-mode {relative_sec,ns} Thrust timestamps absolute nanoseconds (default) or relative seconds
(DIDO only) --plot-elev / --plot-azim for 3D view angles
```

## Example Commands
DRD conversion using existing split text files (auto-detect):
```
python process_drd.py \
  --data-root RAW/drone-racing-dataset/data \
  --out-dir IIT-DRD \
  --plots --overwrite
```

DIDO conversion with random split & plots:
```
python process_dido.py \
  --data-root RAW/DidoRaw \
  --out-dir DIDO \
  --plots --split-ratios 0.6 0.2 0.2 --seed 7
```

Explicit lists:
```
python process_drd.py --train-list train.txt --eval-list eval.txt --test-list test.txt --plots
python process_dido.py --train-list train.txt --eval-list val.txt --test-list test.txt --plots
```

## Output Consistency
- groundTruthPoses.csv: consistent header across both datasets.
- IMU files use relative seconds; thrust timestamps differ (DRD = ns, DIDO = relative seconds) reflecting original data semantics.

## Progress Bar
Both scripts show a tqdm bar unless `--no-progress` is specified.

## Notes / Future Ideas
- Optional parallelization of per-flight/sequence conversion.
- Unified thrust timestamp convention (currently differs intentionally).
- Additional metadata summaries (JSON) per split.
