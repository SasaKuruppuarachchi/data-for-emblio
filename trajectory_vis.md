# trajectory_vis.py Documentation

`trajectory_vis.py` renders static PNG plots and/or animated GIFs for any processed sequence directory containing a `groundTruthPoses.csv` file (columns: `timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z`). It supports 2D XY plotting, optional side‑by‑side 3D view, body-frame axes rendering, and animated progressive drawing with timestamps.

## Quick Start
```
python trajectory_vis.py --sequence-path IIT-DRD/train/flight-02p-ellipse
```
Outputs: `trajectory_vis/flight-02p-ellipse_trajectory.png`

Generate GIF (XY only by default):
```
python trajectory_vis.py --sequence-path IIT-DRD/train/flight-02p-ellipse --gif
```
Outputs: `trajectory_gif/flight-02p-ellipse_trajectory.gif`

Enable 3D in GIF:
```
python trajectory_vis.py --sequence-path IIT-DRD/train/flight-02p-ellipse --gif --plots-3d
```

Add body axes (start & end poses only):
```
python trajectory_vis.py --sequence-path IIT-DRD/train/flight-02p-ellipse --axes-mode start_end --axes-scale 0.4
```

Full featured GIF example (3D + axes + custom size + limited frames):
```
python trajectory_vis.py \
  --sequence-path IIT-DRD/train/flight-02p-ellipse \
  --gif --plots-3d --axes-mode decimate --axes-decimate 40 --axes-scale 0.5 \
  --gif-max-frames 180 --gif-fps 12 --gif-opacity-tail 0.3
```

## Output Conventions
- Static images → directory: `trajectory_vis/` by default.
- GIF animations → directory: `trajectory_gif/` by default (unless `--out-dir` is provided).
- Filename pattern: `<sequence_dir_name>_trajectory.(png|gif)`

## CLI Options
| Option | Description | Default |
|--------|-------------|---------|
| `--sequence-path PATH` | Path to sequence directory containing `groundTruthPoses.csv` | (required) |
| `--out-dir PATH` | Override output directory for rendered file | `trajectory_vis` (PNG) or `trajectory_gif` (GIF) |
| `--name NAME` | Override output base filename (without extension) | Sequence dir name |
| `--title TITLE` | Custom plot title (PNG) | Sequence dir name |
| `--dpi INT` | Rendering DPI for PNG/GIF frames | 120 |
| `--plots-3d` | Enable side-by-side 3D subplot | off |
| `--no-3d` | Force disable 3D even if `--plots-3d` was set | off |
| `--elev FLOAT` | 3D elevation angle (degrees) | 35.0 |
| `--azim FLOAT` | 3D azimuth angle (degrees) | -50.0 |
| `--figsize-xy W H` | Figure size for XY subplot (inches) | 5.0 5.0 |
| `--figsize-3d W H` | Size contribution for 3D subplot (inches) when enabled | 5.5 5.0 |
| `--gif` | Generate animated GIF instead of static PNG | off |
| `--gif-fps INT` | GIF frames per second | 15 |
| `--gif-max-frames INT` | Max frames (trajectory downsampled if longer) | 400 |
| `--gif-loop INT` | GIF loop count (0 = infinite) | 0 |
| `--gif-opacity-tail FLOAT` | Opacity (0–1) for earlier path as tail | 0.25 |
| `--gif-skip-3d` | When GIF + 3D requested, skip 3D pane | off |
| `--gif-t-start FLOAT` | Start time (s, relative to first pose timestamp) for GIF window | full start |
| `--gif-t-end FLOAT` | End time (s, relative to first pose timestamp) for GIF window | full end |
| `--axes-mode {none,start_end,decimate,all}` | Draw body axes strategy | none |
| `--axes-scale FLOAT` | Axis length in meters | 0.2 |
| `--axes-decimate INT` | Step interval for `decimate` mode | 25 |
| `--accel-mode {none,start_end,decimate,all}` | Draw gravity-corrected acceleration arrows (magenta) | none |
| `--accel-decimate INT` | Step for accel-mode=decimate | 40 |
| `--accel-scale FLOAT` | Visual scale (meters per 1 m/s^2) | 0.4 |
| `--imu-file PATH` | Override path to `imu_data.csv` | sequence-path/imu_data.csv |
| `--imu-timestamp-mode {auto,relative_sec,ns}` | Force/override IMU timestamp interpretation | auto |
| `--imu-gravity FLOAT` | Gravity magnitude for correction (m/s^2) | 9.80665 |

## Axes Modes
- `none` – No body axes.
- `start_end` – Only first and last pose.
- `decimate` – Every Nth pose (controlled by `--axes-decimate`).
- `all` – Every pose (can be dense and slow for long trajectories).

Axes colors: X (red), Y (green), Z (blue).

## GIF Behavior
- Progressive drawing of XY (and 3D if enabled) paths.
- Past trajectory drawn with reduced opacity (tail) if `--gif-opacity-tail > 0`.
- Fixed XY and 3D axis ranges for stable framing across frames.
- Title line shows elapsed time (seconds) and frame counter: `t=1.234s  |  Frame 10/120`.
- Timestamp units auto-detected: if initial timestamp > 1e12 treated as nanoseconds → converted to seconds.
- Optional time window selection using `--gif-t-start` / `--gif-t-end`; filtering is applied before frame downsampling by `--gif-max-frames`.

## Examples
Minimal PNG with body axes:
```
python trajectory_vis.py \
  --sequence-path DIDO/train/circle/yawForward/some_sequence \
  --axes-mode start_end --axes-scale 0.3
```

High-res PNG with 3D view:
```
python trajectory_vis.py \
  --sequence-path IIT-DRD/train/flight-04a-ellipse \
  --plots-3d --dpi 180 --figsize-xy 6 6 --figsize-3d 6 5
```

Decimated axes GIF (lighter load):
```
python trajectory_vis.py \
  --sequence-path IIT-DRD/train/flight-02p-ellipse \
  --gif --axes-mode decimate --axes-decimate 60 --axes-scale 0.4
```

Full trajectory GIF with 3D and start/end axes only:
```
python trajectory_vis.py \
  --sequence-path IIT-DRD/train/flight-02p-ellipse \
  --gif --plots-3d --axes-mode start_end --axes-scale 0.5 --gif-fps 12
```

Acceleration overlay (start & end, gravity-corrected):
```
python trajectory_vis.py \
  --sequence-path IIT-DRD/train/flight-02p-ellipse \
  --accel-mode start_end --accel-scale 0.5
```

Decimated acceleration + axes in GIF:
```
python trajectory_vis.py \
  --sequence-path IIT-DRD/train/flight-02p-ellipse \
  --gif --plots-3d \
  --axes-mode decimate --axes-decimate 60 --axes-scale 0.4 \
  --accel-mode decimate --accel-decimate 30 --accel-scale 0.6
```

All poses axes GIF (may be heavy):
```
python trajectory_vis.py \
  --sequence-path IIT-DRD/train/flight-02p-ellipse \
  --gif --axes-mode all --axes-scale 0.15 --gif-max-frames 300
```

Subset time window (first 2.5 seconds after 0.5s):
```
python trajectory_vis.py \
  --sequence-path IIT-DRD/train/flight-02p-ellipse \
  --gif --gif-t-start 0.5 --gif-t-end 3.0 --gif-fps 12 --axes-mode start_end
```

## Performance Tips
- Reduce `--gif-max-frames` for very long trajectories.
- Use `--axes-mode decimate` with a larger `--axes-decimate` to keep rendering fast.
- Lower `--dpi` for quicker GIF generation.
- Avoid `--axes-mode all` on trajectories with thousands of poses unless necessary.
- Narrow the time window with `--gif-t-start` / `--gif-t-end` to focus on segments and reduce frames processed.

## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| GIF generation error about imageio | `imageio` missing | `pip install imageio` |
| No axes appear | `--axes-mode` left at default | Add `--axes-mode start_end` (or other) |
| Axes very small/large | Scale mismatch | Adjust `--axes-scale` |
| GIF too large | Too many frames / high DPI | Lower `--gif-max-frames` or `--dpi` |
| Time shows as very large number | Already in seconds | That’s ok; detection uses threshold 1e12 for ns |

## Future Ideas (Not Implemented)
- Per-axis legend / toggle.
- Export MP4.
- Color path by speed or altitude.

---
Generated documentation for `trajectory_vis.py`.
