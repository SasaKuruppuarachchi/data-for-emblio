#!/usr/bin/env python3
"""process_drd.py

Convert the Drone Racing Dataset 500Hz synchronized flight CSV files
(`*_500hz_freq_sync.csv`) into a Blackbird-style dataset layout:

<out_dir>/
  train/<flight_name>/groundTruthPoses.csv
  train/<flight_name>/imu_data.csv
  train/<flight_name>/thrust_data.csv
  eval/<flight_name>/...
  test/<flight_name>/...

Also saves a trajectory plot (XY + optional small 3D oblique) beside the
CSV outputs (flight_trajectory.png by default) unless disabled.

Mapping (from 500Hz CSV header):
  timestamp (microseconds) -> nanoseconds ( * 1000 )
  accel_[x|y|z] -> imu linear acceleration (m/s^2)
  gyro_[x|y|z]  -> imu angular velocity (rad/s)
  thrust[0-3]   -> motor normalized thrust values (0..1) (kept as-is)
  drone_[x|y|z] -> position (m)
  drone_rot[0-8] -> 3x3 rotation matrix row-major -> quaternion (w,x,y,z)
If rotation matrix invalid / not proper SO(3), it is projected to the
nearest valid rotation using SVD before quaternion conversion.

Splits:
  - If --train-list / --eval-list / --test-list provided (each a text file
    with one flight directory name per line) they're used.
  - Else a random split with ratios --split-ratios (default 0.7 0.15 0.15)
    and seed --seed.

Usage example:
  python process_drd.py \
    --data-root RAW/drone-racing-dataset/data \
    --out-dir DRD \
    --overwrite --plots --no-plot-3d

"""
from __future__ import annotations
import argparse
import csv
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # fallback
    def tqdm(x, **kwargs):
        return x

# ---------------- Rotation utilities ---------------- #

def _proj_to_so3(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the closest valid rotation using SVD.
    """
    U, S, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        # Fix improper rotation (reflection)
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj

def rotmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert rotation matrix to quaternion (w, x, y, z).
    Assumes R is (3,3)."""
    R = _proj_to_so3(R)
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        # Find the major diagonal element
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    # Normalize
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q) + 1e-12
    return tuple(q.tolist())

# ---------------- Data structures ---------------- #

@dataclass
class FlightMeta:
    name: str
    csv_path: Path

# ---------------- Splitting logic ---------------- #

def load_list_file(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith('#')]

def random_split(names: List[str], ratios: Tuple[float,float,float], seed: int) -> Dict[str, List[str]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1"
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train+n_val]
    test = shuffled[n_train+n_val:]
    return {"train": train, "eval": val, "test": test}

# ---------------- Plotting ---------------- #

def plot_trajectory(xyz: np.ndarray, out_path: Path, title: str, enable_3d: bool = True, dpi: int = 120) -> None:
    if xyz.size == 0:
        return
    fig = plt.figure(figsize=(6, 5) if enable_3d else (5,5), dpi=dpi)
    if enable_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax_xy = fig.add_subplot(2,1,1)
    else:
        ax_xy = fig.add_subplot(1,1,1)
    ax_xy.plot(xyz[:,0], xyz[:,1], lw=1.0, color='tab:blue')
    ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')
    ax_xy.set_title(title + ' XY')
    ax_xy.axis('equal')
    if enable_3d:
        ax3d = fig.add_subplot(2,1,2, projection='3d')
        ax3d.plot(xyz[:,0], xyz[:,1], xyz[:,2], lw=0.8, color='tab:orange')
        ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
        ax3d.view_init(elev=35, azim=-50)
        ax3d.set_title(title + ' 3D')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ---------------- Core conversion ---------------- #

def convert_flight(flight: FlightMeta, out_dir: Path, overwrite: bool, make_plot: bool, plot_3d: bool, plot_name: str, plot_dpi: int) -> None:
    # Read header first
    with flight.csv_path.open('r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        # We'll parse columns by name -> index
        col_idx = {name: i for i, name in enumerate(header)}
        required = ['timestamp','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','thrust[0]','thrust[1]','thrust[2]','thrust[3]',
                    'drone_x','drone_y','drone_z','drone_rot[0]','drone_rot[1]','drone_rot[2]','drone_rot[3]','drone_rot[4]','drone_rot[5]','drone_rot[6]','drone_rot[7]','drone_rot[8]']
        for r in required:
            if r not in col_idx:
                raise ValueError(f"Missing required column {r} in {flight.csv_path}")
        # Collect rows
        timestamps_ns: List[int] = []
        accel: List[Tuple[float,float,float]] = []
        gyro: List[Tuple[float,float,float]] = []
        thrusts: List[Tuple[float,float,float,float]] = []
        poses: List[Tuple[int,float,float,float,float,float,float,float,float,float]] = []  # (ts_ns, px,py,pz, qw,qx,qy,qz)
        pos_accum: List[Tuple[float,float,float]] = []
        for row in reader:
            if not row:
                continue
            try:
                ts_us = int(row[col_idx['timestamp']])
            except ValueError:
                continue
            ts_ns = ts_us * 1000  # microseconds -> nanoseconds
            ax = float(row[col_idx['accel_x']]); ay = float(row[col_idx['accel_y']]); az = float(row[col_idx['accel_z']])
            gx = float(row[col_idx['gyro_x']]); gy = float(row[col_idx['gyro_y']]); gz = float(row[col_idx['gyro_z']])
            t0 = float(row[col_idx['thrust[0]']]); t1 = float(row[col_idx['thrust[1]']]); t2 = float(row[col_idx['thrust[2]']]); t3 = float(row[col_idx['thrust[3]']])
            px = float(row[col_idx['drone_x']]); py = float(row[col_idx['drone_y']]); pz = float(row[col_idx['drone_z']])
            R = np.array([
                [float(row[col_idx['drone_rot[0]']]), float(row[col_idx['drone_rot[1]']]), float(row[col_idx['drone_rot[2]']])],
                [float(row[col_idx['drone_rot[3]']]), float(row[col_idx['drone_rot[4]']]), float(row[col_idx['drone_rot[5]']])],
                [float(row[col_idx['drone_rot[6]']]), float(row[col_idx['drone_rot[7]']]), float(row[col_idx['drone_rot[8]']])],
            ])
            qw,qx,qy,qz = rotmat_to_quat(R)
            timestamps_ns.append(ts_ns)
            accel.append((ax,ay,az))
            gyro.append((gx,gy,gz))
            thrusts.append((t0,t1,t2,t3))
            poses.append((ts_ns, px,py,pz, qw,qx,qy,qz))
            pos_accum.append((px,py,pz))
    # Prepare output folder
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / 'groundTruthPoses.csv'
    imu_path = out_dir / 'imu_data.csv'
    thrust_path = out_dir / 'thrust_data.csv'
    if not overwrite and (gt_path.exists() or imu_path.exists() or thrust_path.exists()):
        return
    # Write groundTruthPoses.csv
    with gt_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp','p_x','p_y','p_z','q_w','q_x','q_y','q_z'])
        for (ts,px,py,pz,qw,qx,qy,qz) in poses:
            w.writerow([ts, px, py, pz, qw, qx, qy, qz])
    # Write imu_data.csv (timestamp in *seconds* float to match earlier convention)
    t0_ns = timestamps_ns[0] if timestamps_ns else 0
    with imu_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z'])
        for ts,(ax,ay,az),(gx,gy,gz) in zip(timestamps_ns, accel, gyro):
            w.writerow([(ts - t0_ns) / 1e9, ax, ay, az, gx, gy, gz])
    # Write thrust_data.csv (timestamps in nanoseconds to align with groundTruthPoses)
    with thrust_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp','motor0','motor1','motor2','motor3'])
        for ts,(m0,m1,m2,m3) in zip(timestamps_ns, thrusts):
            w.writerow([ts, m0, m1, m2, m3])
    # Plot
    if make_plot and pos_accum:
        xyz = np.array(pos_accum, dtype=float)
        plot_trajectory(xyz, out_dir / plot_name, flight.name, enable_3d=plot_3d, dpi=plot_dpi)

# ---------------- Discovery ---------------- #

def discover_flights(data_root: Path) -> List[FlightMeta]:
    flights: List[FlightMeta] = []
    for split_dir in ['autonomous','piloted']:
        base = data_root / split_dir
        if not base.exists():
            continue
        for d in sorted(base.glob('flight-*')):
            if not d.is_dir():
                continue
            csv_file = d / f"{d.name}_500hz_freq_sync.csv"
            if csv_file.exists():
                flights.append(FlightMeta(name=d.name, csv_path=csv_file))
    return flights

# ---------------- Main ---------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Convert drone-racing-dataset 500Hz CSVs to Blackbird-like format")
    p.add_argument('--data-root', type=Path, default=Path('RAW/drone-racing-dataset/data'), help='Root containing autonomous/ and piloted/')
    p.add_argument('--out-dir', type=Path, default=Path('IIT-DRD'), help='Output dataset root')
    p.add_argument('--train-list', type=Path, help='Optional train list (overrides auto)')
    p.add_argument('--eval-list', type=Path, help='Optional eval list (overrides auto)')
    p.add_argument('--test-list', type=Path, help='Optional test list (overrides auto)')
    p.add_argument('--split-ratios', type=float, nargs=3, default=(0.7,0.15,0.15), help='Ratios if no list files found')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--plots', action='store_true', help='Generate trajectory plots')
    p.add_argument('--no-plot-3d', action='store_true')
    p.add_argument('--plot-name', default='flight_trajectory.png')
    p.add_argument('--plot-dpi', type=int, default=120)
    p.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    return p.parse_args()

# Helper to auto-detect list files (train.txt, eval.txt, test.txt) located at data-root parent folder or within data-root

def autodetect_lists(data_root: Path) -> Dict[str, Path]:
    candidates = {}
    search_dirs = [data_root.parent, data_root]
    for label, fname in [('train','train.txt'), ('eval','eval.txt'), ('test','test.txt')]:
        for base in search_dirs:
            f = base / fname
            if f.exists():
                candidates[label] = f
                break
    return candidates

def main():
    args = parse_args()
    flights = discover_flights(args.data_root)
    if not flights:
        print("No flights discovered.")
        return
    name_to_meta = {f.name: f for f in flights}
    all_names = list(name_to_meta.keys())

    # Decide splits
    lists_used = {}
    auto_lists = autodetect_lists(args.data_root)
    if any([args.train_list, args.eval_list, args.test_list]):
        train = load_list_file(args.train_list) if args.train_list else []
        val = load_list_file(args.eval_list) if args.eval_list else []
        test = load_list_file(args.test_list) if args.test_list else []
        lists_used['explicit'] = True
    elif len(auto_lists) == 3:
        train = load_list_file(auto_lists['train'])
        val = load_list_file(auto_lists['eval'])
        test = load_list_file(auto_lists['test'])
        lists_used['auto'] = {k: str(v) for k,v in auto_lists.items()}
    else:
        splits = random_split(all_names, tuple(args.split_ratos) if hasattr(args,'split_ratos') else tuple(args.split_ratios), args.seed)
        train, val, test = splits['train'], splits['eval'], splits['test']
        lists_used['random'] = args.split_ratios

    # Build unified work list for progress bar
    work_items = []
    for split_name, names in [('train', train), ('eval', val), ('test', test)]:
        for n in names:
            work_items.append((split_name, n))

    iterator = work_items
    if not args.no_progress:
        iterator = tqdm(work_items, desc='Processing flights', unit='flight')

    processed = 0
    for split_name, name in iterator:
        meta = name_to_meta.get(name)
        if not meta:
            continue
        out_dir = args.out_dir / split_name / name
        convert_flight(meta, out_dir, overwrite=args.overwrite, make_plot=args.plots, plot_3d=not args.no_plot_3d, plot_name=args.plot_name, plot_dpi=args.plot_dpi)
        processed += 1

    print(f"Done. Flights processed: {processed}")
    if lists_used:
        print("Split source:", lists_used)

if __name__ == '__main__':
    main()
