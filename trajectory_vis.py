#!/usr/bin/env python3
"""trajectory_vis.py

Standalone trajectory visualization (static PNG and optional animated GIF)
for any processed sequence folder produced by process_drd.py or
process_dido.py.

Given a path to a sequence directory containing groundTruthPoses.csv
(columns: timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z), this script renders
an XY plot plus optional oblique 3D plot mimicking the styles used in
the dataset conversion scripts.

Usage examples (PNG):
  python trajectory_vis.py --sequence-path IIT-DRD/train/flight-04a-ellipse \
	  --out-dir trajectory_vis --plots-3d

	python trajectory_vis.py --sequence-path DIDO/train/circle/yawForward/<seq> \
			--title MySequence --no-3d

Animated GIF example:
	python trajectory_vis.py --sequence-path IIT-DRD/train/flight-04a-ellipse \
			--gif --gif-fps 20 --gif-max-frames 300

Defaults:
	- Static PNG output directory: ./trajectory_vis (unless --out-dir supplied)
	- GIF output directory: ./trajectory_gif (unless --out-dir supplied when --gif used)
	- Output filenames: <sequence_dir_name>_trajectory.(png|gif)

Exit codes:
  0 on success, non-zero on failure (e.g. missing file).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import csv
from typing import List, Tuple

import numpy as np
try:
	import imageio.v2 as imageio  # prefer v2 api
except Exception:  # fallback stub
	imageio = None
import matplotlib
matplotlib.use('Agg')  # headless safe
import matplotlib.pyplot as plt


def load_ground_truth(gt_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Load groundTruthPoses.csv returning positions (N,3), timestamps (N,), quats (N,4).

	Quaternions expected as q_w,q_x,q_y,q_z. Returns empty arrays if load fails.
	"""
	if not gt_path.exists():
		return np.empty((0,3)), np.empty((0,), dtype=np.int64), np.empty((0,4))
	positions: List[Tuple[float,float,float]] = []
	timestamps: List[int] = []
	quats: List[Tuple[float,float,float,float]] = []
	with gt_path.open('r', newline='') as f:
		reader = csv.reader(f)
		header = next(reader, None)
		# Flexible: accept either exact header or any with p_x at index 1
		for row in reader:
			if not row or len(row) < 8:
				continue
			try:
				ts = int(row[0])
				px = float(row[1]); py = float(row[2]); pz = float(row[3])
				qw = float(row[4]); qx = float(row[5]); qy = float(row[6]); qz = float(row[7])
			except ValueError:
				continue
			timestamps.append(ts)
			positions.append((px,py,pz))
			quats.append((qw,qx,qy,qz))
	if not positions:
		return np.empty((0,3)), np.empty((0,), dtype=np.int64), np.empty((0,4))
	return (np.asarray(positions, dtype=float),
			np.asarray(timestamps, dtype=np.int64),
			np.asarray(quats, dtype=float))


def load_imu(imu_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Load imu_data.csv returning timestamps, accel (N,3), gyro (N,3).

	Expected columns header containing: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
	Timestamp may be float seconds-from-start or integer nanoseconds.
	Returns empty arrays on failure.
	"""
	if not imu_path or not imu_path.exists():
		return np.empty((0,), dtype=float), np.empty((0,3)), np.empty((0,3))
	ts_list: List[float] = []
	acc_list: List[Tuple[float,float,float]] = []
	gyr_list: List[Tuple[float,float,float]] = []
	with imu_path.open('r', newline='') as f:
		reader = csv.reader(f)
		header = next(reader, None)
		for row in reader:
			if len(row) < 7:
				continue
			try:
				ts_raw = row[0].strip()
				if ts_raw.isdigit():
					# nanoseconds integer string
					ts_val = float(int(ts_raw))
				else:
					# assume float seconds
					ts_val = float(ts_raw)
				ax = float(row[1]); ay = float(row[2]); az = float(row[3])
				gx = float(row[4]); gy = float(row[5]); gz = float(row[6])
			except ValueError:
				continue
			ts_list.append(ts_val)
			acc_list.append((ax,ay,az))
			gyr_list.append((gx,gy,gz))
	if not acc_list:
		return np.empty((0,), dtype=float), np.empty((0,3)), np.empty((0,3))
	return (np.asarray(ts_list, dtype=float),
			np.asarray(acc_list, dtype=float),
			np.asarray(gyr_list, dtype=float))

def infer_imu_timestamp_mode(ts: np.ndarray) -> str:
	"""Infer whether IMU timestamps are relative seconds or nanoseconds as float-int stored.

	If all values very large (>1e12) treat as nanoseconds. Otherwise assume seconds.
	"""
	if ts.size == 0:
		return 'relative_sec'
	if np.median(ts) > 1e12:
		return 'ns'
	return 'relative_sec'

def align_accel_with_gt(imu_ts: np.ndarray, acc: np.ndarray, gt_ts: np.ndarray, gt_quats: np.ndarray,
						 timestamp_mode: str) -> Tuple[np.ndarray, np.ndarray]:
	"""Interpolate/associate acceleration samples to ground-truth pose indices.

	Returns (acc_world_no_g, valid_mask) where acc_world_no_g shape is (N,3).
	If no IMU, returns zeros.
	"""
	N = gt_ts.shape[0]
	if N == 0 or imu_ts.size == 0:
		return np.zeros((N,3), dtype=float), np.zeros((N,), dtype=bool)
	# Convert imu timestamps to nanoseconds if needed for alignment
	if timestamp_mode == 'relative_sec':
		# Reconstruct relative seconds to nanoseconds using gt start reference
		gt0_ns = gt_ts[0]
		imu_ts_ns = gt0_ns + (imu_ts * 1e9)
	else:  # 'ns'
		imu_ts_ns = imu_ts
	# For each gt_ts, find nearest imu sample (binary search)
	acc_out = np.zeros((N,3), dtype=float)
	valid = np.zeros((N,), dtype=bool)
	imu_ts_ns = imu_ts_ns.astype(np.int64)
	for i, t in enumerate(gt_ts):
		# np.searchsorted to find insertion
		idx = np.searchsorted(imu_ts_ns, t)
		candidates = []
		if idx < imu_ts_ns.size:
			candidates.append(idx)
		if idx > 0:
			candidates.append(idx-1)
		if not candidates:
			continue
		# choose nearest in time
		best = min(candidates, key=lambda k: abs(int(imu_ts_ns[k]) - int(t)))
		acc_body = acc[best]
		# Rotate body accel to world using quaternion
		R = quat_to_rot(gt_quats[i])  # world_from_body
		acc_world = R @ acc_body
		acc_out[i] = acc_world
		valid[i] = True
	return acc_out, valid

def gravity_correct(acc_world: np.ndarray, gravity_mag: float) -> np.ndarray:
	"""Remove gravity from world-frame acceleration.

	Assumes world Z axis is vertical. We auto-detect sign: if the median Z component
	|median_z| is within [0.5*g, 1.5*g], we treat its sign as the gravity direction and subtract it.
	Otherwise we assume accelerometer already had gravity removed and return unchanged.
	"""
	if acc_world.size == 0:
		return acc_world
	median_z = float(np.median(acc_world[:,2]))
	if 0.5*gravity_mag < abs(median_z) < 1.5*gravity_mag:
		g_vec = np.array([0.0, 0.0, np.sign(median_z)*gravity_mag])
		return acc_world - g_vec
	# fallback: previous incorrect assumption produced inflated values; no correction applied
	return acc_world

def _draw_accel_arrows_2d(ax, xyz: np.ndarray, acc_w_no_g: np.ndarray, mode: str, decimate: int, scale: float):
	indices = list(_iter_axis_indices(len(xyz), mode, decimate))
	for i in indices:
		vec = acc_w_no_g[i]
		if not np.isfinite(vec).all():
			continue
		base = xyz[i,:2]
		arrow = vec[:2] * scale
		end = base + arrow
		ax.annotate('', xy=end, xytext=base, arrowprops=dict(arrowstyle='->', color='magenta', lw=1.6, alpha=0.85))

def _draw_accel_arrows_3d(ax, xyz: np.ndarray, acc_w_no_g: np.ndarray, mode: str, decimate: int, scale: float):
	indices = list(_iter_axis_indices(len(xyz), mode, decimate))
	for i in indices:
		vec = acc_w_no_g[i]
		if not np.isfinite(vec).all():
			continue
		base = xyz[i]
		end = base + vec * scale
		ax.plot([base[0], end[0]],[base[1], end[1]],[base[2], end[2]], color='magenta', lw=1.5, alpha=0.85)



def plot_trajectory(xyz: np.ndarray, quats: np.ndarray, out_file: Path, title: str, enable_3d: bool, dpi: int,
					elev: float, azim: float, figsize_xy: Tuple[float,float], figsize_3d: Tuple[float,float],
					axes_mode: str, axes_scale: float, axes_decimate: int,
					acc_mode: str='none', acc_decimate: int=40, acc_scale: float=0.4,
					acc_world_no_g: np.ndarray | None = None) -> bool:
	if xyz.size == 0:
		return False
	if enable_3d:
		fig = plt.figure(figsize=(figsize_xy[0]+figsize_3d[0], max(figsize_xy[1], figsize_3d[1])), dpi=dpi)
		gs = fig.add_gridspec(1,2, width_ratios=[1,1.05])
		ax_xy = fig.add_subplot(gs[0,0])
		ax3d = fig.add_subplot(gs[0,1], projection='3d')
	else:
		fig, ax_xy = plt.subplots(figsize=figsize_xy, dpi=dpi)
		ax3d = None
	# 2D
	ax_xy.plot(xyz[:,0], xyz[:,1], lw=1.0, color='tab:blue')
	ax_xy.scatter([xyz[0,0]],[xyz[0,1]], c='green', s=25, label='start')
	ax_xy.scatter([xyz[-1,0]],[xyz[-1,1]], c='red', s=25, label='end')
	ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')
	ax_xy.set_title(f"{title} XY")
	ax_xy.axis('equal')
	ax_xy.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
	ax_xy.legend(loc='best', fontsize=8)
	if axes_mode != 'none':
		_draw_axes_2d(ax_xy, xyz, quats, axes_mode, axes_scale, axes_decimate)
	if acc_mode != 'none' and acc_world_no_g is not None and acc_world_no_g.shape[0] == xyz.shape[0]:
		_draw_accel_arrows_2d(ax_xy, xyz, acc_world_no_g, acc_mode, acc_decimate, acc_scale)
	if ax3d is not None:
		ax3d.plot(xyz[:,0], xyz[:,1], xyz[:,2], lw=0.8, color='tab:orange')
		ax3d.scatter([xyz[0,0]],[xyz[0,1]],[xyz[0,2]], c='green', s=18)
		ax3d.scatter([xyz[-1,0]],[xyz[-1,1]],[xyz[-1,2]], c='red', s=18)
		ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
		ax3d.view_init(elev=elev, azim=azim)
		ax3d.set_title('3D')
		if axes_mode != 'none':
			_draw_axes_3d(ax3d, xyz, quats, axes_mode, axes_scale, axes_decimate)
		if acc_mode != 'none' and acc_world_no_g is not None and acc_world_no_g.shape[0] == xyz.shape[0]:
			_draw_accel_arrows_3d(ax3d, xyz, acc_world_no_g, acc_mode, acc_decimate, acc_scale)
	# Save figure
	out_file.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_file)
	plt.close(fig)
	return True
def quat_to_rot(q: np.ndarray) -> np.ndarray:
	qw,qx,qy,qz = q
	# Normalize
	n = np.linalg.norm(q)
	if n == 0:
		return np.eye(3)
	qw,qx,qy,qz = q / n
	# Rotation matrix
	return np.array([
		[1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
		[    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
		[    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
	], dtype=float)

def _iter_axis_indices(n: int, mode: str, decimate: int):
	if mode == 'start_end':
		if n >= 1:
			yield 0
		if n >= 2:
			yield n-1
	elif mode == 'decimate':
		step = max(decimate,1)
		for i in range(0, n, step):
			yield i
	elif mode == 'all':
		for i in range(n):
			yield i
	# 'none' -> no yield

def _draw_axes_2d(ax, xyz, quats, mode: str, scale: float, decimate: int):
	n = len(xyz)
	if n == 0 or quats.shape[0] != n:
		return
	colors = ['#d62728','#2ca02c','#1f77b4']  # x=red, y=green, z=blue
	indices = list(_iter_axis_indices(n, mode, decimate))
	for i in indices:
		R = quat_to_rot(quats[i])
		origin = xyz[i,:2]
		is_last = (i == indices[-1])
		is_first = (i == indices[0])
		highlight = is_last or is_first
		if mode == 'decimate':
			lw_main = 2.2 if highlight else 0.9
			alpha_main = 0.97 if highlight else 0.35
		else:
			lw_main = 2.2 if highlight else 1.2
			alpha_main = 0.95 if highlight else 0.8
		for j in range(3):
			vec = R[:2, j] * scale
			end = origin + vec
			ax.plot([origin[0], end[0]],[origin[1], end[1]], color=colors[j], linewidth=lw_main, alpha=alpha_main, solid_capstyle='round', zorder=5)
			# tiny head (triangle) for clarity
			head_scale = (0.11 if is_last else 0.07) * scale
			perp = np.array([ -vec[1], vec[0] ])
			if np.linalg.norm(perp) > 1e-9:
				perp = perp / np.linalg.norm(perp)
				head_base = end - vec * 0.15
				p1 = end
				p2 = head_base + perp * head_scale
				p3 = head_base - perp * head_scale
				ax.fill([p1[0],p2[0],p3[0]],[p1[1],p2[1],p3[1]], color=colors[j], alpha=alpha_main, zorder=6)

def _draw_axes_3d(ax, xyz, quats, mode: str, scale: float, decimate: int):
	n = len(xyz)
	if n == 0 or quats.shape[0] != n:
		return
	colors = ['#d62728','#2ca02c','#1f77b4']
	indices = list(_iter_axis_indices(n, mode, decimate))
	for i in indices:
		R = quat_to_rot(quats[i])
		o = xyz[i]
		is_last = (i == indices[-1])
		is_first = (i == indices[0])
		highlight = is_last or is_first
		if mode == 'decimate':
			lw_main = 2.0 if highlight else 0.8
			alpha_main = 0.9 if highlight else 0.32
		else:
			lw_main = 2.0 if highlight else 1.3
			alpha_main = 0.9 if highlight else 0.8
		for j in range(3):
			v = R[:,j] * scale
			ax.plot([o[0], o[0]+v[0]], [o[1], o[1]+v[1]], [o[2], o[2]+v[2]], color=colors[j], linewidth=lw_main, alpha=alpha_main)


def parse_args():
	p = argparse.ArgumentParser(description="Visualize a processed trajectory (groundTruthPoses.csv)")
	p.add_argument('--sequence-path', required=True, type=Path,
				   help='Path to sequence directory containing groundTruthPoses.csv')
	p.add_argument('--out-dir', type=Path, default=Path('trajectory_vis'),
				   help='Directory to store rendered plot (default: trajectory_vis)')
	p.add_argument('--name', type=str, default=None,
				   help='Custom output image base name (default: sequence dir name)')
	p.add_argument('--title', type=str, default=None,
				   help='Custom plot title (default: sequence dir name)')
	p.add_argument('--dpi', type=int, default=120)
	p.add_argument('--plots-3d', action='store_true', help='Enable side-by-side 3D view')
	p.add_argument('--no-3d', action='store_true', help='Force disable 3D even if plots-3d specified later')
	p.add_argument('--elev', type=float, default=35.0, help='3D elevation angle')
	p.add_argument('--azim', type=float, default=-50.0, help='3D azimuth angle')
	p.add_argument('--figsize-xy', type=float, nargs=2, default=(5.0,5.0), help='XY figure size (w h)')
	p.add_argument('--figsize-3d', type=float, nargs=2, default=(5.5,5.0), help='3D subplot size contribution (w h) when enabled')
	# GIF options
	p.add_argument('--gif', action='store_true', help='Generate animated GIF (XY; add --plots-3d for side 3D unless --gif-skip-3d)')
	p.add_argument('--gif-fps', type=int, default=15, help='Frames per second for GIF')
	p.add_argument('--gif-max-frames', type=int, default=400, help='Max frames (downsample if trajectory longer)')
	p.add_argument('--gif-loop', type=int, default=0, help='GIF loop count (0=infinite)')
	p.add_argument('--gif-opacity-tail', type=float, default=0.25, help='Opacity for past path (0-1)')
	p.add_argument('--gif-t-start', type=float, default=None, help='Start time (seconds, relative to first timestamp) to begin GIF (inclusive)')
	p.add_argument('--gif-t-end', type=float, default=None, help='End time (seconds, relative to first timestamp) to stop GIF (inclusive)')
	p.add_argument('--gif-skip-3d', action='store_true', help='Do not include 3D even if --plots-3d set when generating GIF')
	p.add_argument('--no-progress', action='store_true', help='Disable textual progress bar for GIF rendering')
	# Axes drawing
	p.add_argument('--axes-mode', choices=['none','start_end','decimate','all'], default='none',
				   help='Draw body axes: none, start_end, decimate, all')
	p.add_argument('--axes-scale', type=float, default=0.2, help='Axis length in meters')
	p.add_argument('--axes-decimate', type=int, default=25, help='Step when mode=decimate')
	# Acceleration (IMU) overlay options
	p.add_argument('--accel-mode', choices=['none','start_end','decimate','all'], default='none',
			   help='Draw gravity-corrected acceleration arrows (magenta) at poses using matching mode semantics')
	p.add_argument('--accel-decimate', type=int, default=40, help='Step when accel-mode=decimate')
	p.add_argument('--accel-scale', type=float, default=0.4, help='Scaling factor (visual meters per 1 m/s^2) for acceleration arrow length')
	p.add_argument('--imu-file', type=Path, default=None, help='Optional explicit path to imu_data.csv (otherwise sequence-path/imu_data.csv)')
	p.add_argument('--imu-timestamp-mode', choices=['auto','relative_sec','ns'], default='auto',
			   help='Hint for interpreting imu timestamps when disambiguation is needed (auto tries to infer)')
	p.add_argument('--imu-gravity', type=float, default=9.80665, help='Gravity magnitude assumed for correction (m/s^2)')
	return p.parse_args()


def _make_gif(xyz: np.ndarray, quats: np.ndarray, ts: np.ndarray, out_file: Path, fps: int, max_frames: int, loop: int,
	opacity_tail: float, axes_mode: str, axes_scale: float, axes_decimate: int,
	enable_3d: bool, elev: float, azim: float, figsize_xy: Tuple[float,float], figsize_3d: Tuple[float,float],
	show_progress: bool, t_start: float | None = None, t_end: float | None = None,
	acc_mode: str='none', acc_decimate: int=40, acc_scale: float=0.4, acc_world_no_g: np.ndarray | None = None) -> bool:
	if imageio is None:
		print('[ERROR] imageio not available, cannot create GIF')
		return False
	n = len(xyz)
	if n < 2:
		return False
	# Determine frame indices (evenly spaced) over full range first
	if n <= max_frames:
		base_idxs = np.arange(n)
	else:
		base_idxs = np.linspace(0, n - 1, max_frames).astype(int)

	# Time window filtering (convert provided seconds into same units as normalized seconds we derive later)
	t0_raw = ts[0] if len(ts) else 0
	divisor_probe = 1e9 if t0_raw > 1e12 else 1.0
	if t_start is not None or t_end is not None:
		# compute per-index seconds relative to start
		sec_rel = (ts[base_idxs] - t0_raw) / divisor_probe
		mask = np.ones_like(base_idxs, dtype=bool)
		if t_start is not None:
			mask &= sec_rel >= t_start - 1e-9
		if t_end is not None:
			mask &= sec_rel <= t_end + 1e-9
		idxs = base_idxs[mask]
		if len(idxs) == 0:
			print('[WARN] Time window produced no frames; aborting GIF')
			return False
	else:
		idxs = base_idxs
	frames = []
	# Precompute extents for stable axes
	minx, maxx = np.min(xyz[:,0]), np.max(xyz[:,0])
	miny, maxy = np.min(xyz[:,1]), np.max(xyz[:,1])
	minz, maxz = np.min(xyz[:,2]), np.max(xyz[:,2])
	dx = max(maxx - minx, 1e-6)
	dy = max(maxy - miny, 1e-6)
	dz = max(maxz - minz, 1e-6)
	# unified scale padding
	pad_ratio = 0.07
	pad_x = pad_ratio * dx
	pad_y = pad_ratio * dy
	pad_z = pad_ratio * dz
	# For 3D equal aspect cube extents
	if enable_3d:
		max_range = max(dx, dy, dz)
		# Center each axis
		cx = 0.5 * (minx + maxx)
		cy = 0.5 * (miny + maxy)
		cz = 0.5 * (minz + maxz)
		half = 0.5 * max_range * (1 + 2*pad_ratio)
		fixed_xlim = (cx - half, cx + half)
		fixed_ylim = (cy - half, cy + half)
		fixed_zlim = (cz - half, cz + half)
	else:
		fixed_xlim = (minx - pad_x, maxx + pad_x)
		fixed_ylim = (miny - pad_y, maxy + pad_y)
		fixed_zlim = (minz - pad_z, maxz + pad_z)
	# Normalize timestamps to start at zero (assume ns if large)
	t0 = ts[0] if len(ts) else 0
	# Heuristic: if magnitude > 1e12 treat as nanoseconds
	divisor = 1e9 if t0 > 1e12 else 1.0
	start_time = None
	import time
	for i, idx in enumerate(idxs):
		if start_time is None:
			start_time = time.time()
		if enable_3d:
			fig = plt.figure(figsize=(figsize_xy[0]+figsize_3d[0], max(figsize_xy[1], figsize_3d[1])), dpi=100)
			gs = fig.add_gridspec(1,2, width_ratios=[1,1.05])
			ax = fig.add_subplot(gs[0,0])
			ax3d = fig.add_subplot(gs[0,1], projection='3d')
		else:
			fig, ax = plt.subplots(figsize=figsize_xy, dpi=100)
			ax3d = None
		# Tail path faint
		if opacity_tail > 0 and idx > 1:
			ax.plot(xyz[:idx+1,0], xyz[:idx+1,1], color='tab:blue', alpha=opacity_tail, lw=1.0)
		# Current progressive path solid
		ax.plot(xyz[:idx+1,0], xyz[:idx+1,1], color='tab:blue', lw=1.4)
		ax.scatter([xyz[0,0]],[xyz[0,1]], c='green', s=30)
		ax.scatter([xyz[idx,0]],[xyz[idx,1]], c='red', s=24)
		# Axes for current frame only (lighter weight) if enabled
		if axes_mode != 'none':
			_draw_axes_2d(ax, xyz[:idx+1], quats[:idx+1], 'start_end' if axes_mode=='start_end' else ('decimate' if axes_mode=='decimate' else axes_mode), axes_scale, axes_decimate)
		if acc_mode != 'none' and acc_world_no_g is not None:
			_draw_accel_arrows_2d(ax, xyz[:idx+1], acc_world_no_g[:idx+1], 'start_end' if acc_mode=='start_end' else ('decimate' if acc_mode=='decimate' else acc_mode), acc_decimate, acc_scale)
		ax.set_xlim(*fixed_xlim)
		ax.set_ylim(*fixed_ylim)
		ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
		elapsed_sec = (ts[idx]-t0)/divisor if len(ts) else 0.0
		ax.set_title(f't={elapsed_sec:0.3f}s  |  Frame {i+1}/{len(idxs)}')
		# Re-apply fixed limits (matplotlib sometimes autos-scales on new artists)
		ax.set_xlim(*fixed_xlim)
		ax.set_ylim(*fixed_ylim)
		ax.set_aspect('equal', adjustable='box')
		ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
		if enable_3d and ax3d is not None:
			# 3D progressive path
			ax3d.plot(xyz[:idx+1,0], xyz[:idx+1,1], xyz[:idx+1,2], lw=0.8, color='tab:orange')
			ax3d.scatter([xyz[0,0]],[xyz[0,1]],[xyz[0,2]], c='green', s=18)
			ax3d.scatter([xyz[idx,0]],[xyz[idx,1]],[xyz[idx,2]], c='red', s=18)
			if axes_mode != 'none':
				_draw_axes_3d(ax3d, xyz[:idx+1], quats[:idx+1], 'start_end' if axes_mode=='start_end' else ('decimate' if axes_mode=='decimate' else axes_mode), axes_scale, axes_decimate)
			if acc_mode != 'none' and acc_world_no_g is not None:
				_draw_accel_arrows_3d(ax3d, xyz[:idx+1], acc_world_no_g[:idx+1], 'start_end' if acc_mode=='start_end' else ('decimate' if acc_mode=='decimate' else acc_mode), acc_decimate, acc_scale)
			ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
			ax3d.view_init(elev=elev, azim=azim)
			# Fixed ranges
			ax3d.set_xlim(*fixed_xlim)
			ax3d.set_ylim(*fixed_ylim)
			ax3d.set_zlim(*fixed_zlim)
			ax3d.set_title('3D')
		fig.tight_layout()
		frames.append(_fig_to_array(fig))
		if show_progress:
			elapsed = time.time() - start_time
			done = i + 1
			frac = done / len(idxs)
			eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
			bar_len = 28
			filled = int(bar_len * frac)
			bar = '#' * filled + '-' * (bar_len - filled)
			fps_eff = done / elapsed if elapsed > 0 else 0.0
			msg = f"[GIF] |{bar}| {done}/{len(idxs)} {frac*100:5.1f}%  frame_t={elapsed/done:0.3f}s  fps~{fps_eff:0.1f} ETA {eta:0.1f}s"
			print('\r' + msg, end='', flush=True)
	if show_progress:
		print()  # newline after bar
		plt.close(fig)
	imageio.mimsave(out_file, frames, fps=fps, loop=loop)
	return True

def _fig_to_array(fig):
	fig.canvas.draw()
	# Use renderer buffer (RGBA) then drop alpha
	w, h = fig.canvas.get_width_height()
	buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
	arr = buf.reshape((h, w, 4))[:, :, :3]
	return arr


def main():
	args = parse_args()
	seq_dir = args.sequence_path
	gt_path = seq_dir / 'groundTruthPoses.csv'
	if not gt_path.exists():
		print(f"[ERROR] groundTruthPoses.csv not found in {seq_dir}")
		sys.exit(2)
	xyz, ts, quats = load_ground_truth(gt_path)
	if xyz.size == 0:
		print(f"[ERROR] No valid pose rows in {gt_path}")
		sys.exit(3)
	# Optional IMU acceleration overlay
	acc_world_no_g = None
	if args.accel_mode != 'none':
		imu_path = args.imu_file if args.imu_file else (seq_dir / 'imu_data.csv')
		imu_ts_raw, acc_body, gyr = load_imu(imu_path)
		if imu_ts_raw.size == 0:
			print(f"[WARN] Acceleration overlay requested but no IMU data at {imu_path}")
		else:
			mode = args.imu_timestamp_mode
			if mode == 'auto':
				mode = infer_imu_timestamp_mode(imu_ts_raw)
			acc_w, valid_mask = align_accel_with_gt(imu_ts_raw, acc_body, ts, quats, mode)
			if valid_mask.any():
				# gravity correction
				acc_world_no_g = gravity_correct(acc_w, args.imu_gravity)
			else:
				print('[WARN] No valid IMU associations for acceleration overlay')
	title = args.title if args.title else seq_dir.name
	out_base = args.name if args.name else seq_dir.name
	# Decide default out dir: if gif requested and out-dir not explicitly set by user (heuristic: value is default) use trajectory_gif
	default_png_dir = Path('trajectory_vis')
	default_gif_dir = Path('trajectory_gif')
	user_supplied_out = args.out_dir != default_png_dir  # if user changed from default
	effective_out_dir = args.out_dir
	if args.gif and not user_supplied_out and str(args.out_dir) == str(default_png_dir):
		effective_out_dir = default_gif_dir
	out_file = effective_out_dir / f"{out_base}_trajectory.{'gif' if args.gif else 'png'}"
	enable_3d = (args.plots_3d and not args.no_3d)
	if args.gif:
		# Always ensure output dir exists
		out_file.parent.mkdir(parents=True, exist_ok=True)
		gif_ok = _make_gif(xyz, quats, ts, out_file, args.gif_fps, args.gif_max_frames, args.gif_loop, args.gif_opacity_tail,
				   args.axes_mode, args.axes_scale, args.axes_decimate, enable_3d and not args.gif_skip_3d,
				   args.elev, args.azim, tuple(args.figsize_xy), tuple(args.figsize_3d), not args.no_progress,
				   args.gif_t_start, args.gif_t_end,
				   acc_mode=args.accel_mode, acc_decimate=args.accel_decimate, acc_scale=args.accel_scale, acc_world_no_g=acc_world_no_g)
		if not gif_ok:
			print('[ERROR] GIF generation failed')
			sys.exit(5)
		print(f"[OK] Wrote {out_file}")
		# Also optionally emit a static PNG if user wants (implicit when plots-3d or no-3d set?) -> skip for simplicity unless separately requested
	else:
		ok = plot_trajectory(xyz, quats, out_file, title, enable_3d, args.dpi, args.elev, args.azim,
				 tuple(args.figsize_xy), tuple(args.figsize_3d), args.axes_mode, args.axes_scale, args.axes_decimate,
				 acc_mode=args.accel_mode, acc_decimate=args.accel_decimate, acc_scale=args.accel_scale, acc_world_no_g=acc_world_no_g)
		if not ok:
			print("[ERROR] Plot failed (empty trajectory)")
			sys.exit(4)
		print(f"[OK] Wrote {out_file}")


if __name__ == '__main__':
	main()

