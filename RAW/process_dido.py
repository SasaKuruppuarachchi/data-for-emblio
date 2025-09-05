"""process_dido.py

Convert the DIDO raw HDF5 sequences into a Blackbird‑style dataset layout
with an interface aligned to process_drd.py for consistency.

Output layout (default root: sibling ../DIDO unless --out-dir absolute):

<out_dir>/
	train/<pattern>/yawForward/<seq>/groundTruthPoses.csv
	train/<pattern>/yawForward/<seq>/imu_data.csv
	train/<pattern>/yawForward/<seq>/thrust_data.csv
	eval/...
	test/...

Mapping assumptions:
	- Raw sequences under <data-root>/<sequence_dir>/data.hdf5 (default data-root = script_dir/DidoRaw)
	- HDF5 datasets:
			ts (N,) float64 UNIX epoch seconds OR micro/nano → detected & normalized
			acc (N,3) linear acceleration m/s^2
			gyr (N,3) angular velocity rad/s
			gt_p (N,3) position (m)
			gt_q (N,4) quaternion (w,x,y,z)
			meas_rpm (N,>=5) first column time-like (ignored) next 4 columns → thrust_1..4 (fallback zeros otherwise)
	- groundTruthPoses.csv columns: timestamp_ns(int), p_x,y,z, q_w,q_x,q_y,q_z
	- imu_data.csv columns: timestamp_sec(float, start=0), acc_x,y,z, gyro_x,y,z
	- thrust_data.csv columns: timestamp_sec(float, start=0), thrust_1..4

Splits:
	- If explicit list files supplied via --train-list/--eval-list/--test-list they are used.
	- Else if train.txt / val.txt(or eval.txt) / test.txt exist in data-root they are auto‑loaded.
	- Else a random split with --split-ratios (default 0.7 0.15 0.15) & --seed.
	- val -> eval mapping for output folder name.

Key parity with process_drd.py:
	- Similar CLI (data-root, out-dir, list overrides, split-ratios, overwrite, plotting, progress bar)
	- Progress bar across all sequences (disable with --no-progress)

Usage example:
	python process_dido.py \
		--data-root ./DidoRaw \
		--out-dir DIDO/BlackbirdLike \
		--plots --overwrite

This script is idempotent; existing sequence folders are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse, csv, random, sys, signal, os
from pathlib import Path
from typing import List, Tuple, Dict

import h5py, numpy as np, matplotlib
matplotlib.use('Agg')  # headless safe
import matplotlib.pyplot as plt
try:
	from tqdm import tqdm
except Exception:  # fallback
	def tqdm(x, **kwargs):
		return x
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "DidoRaw"
DEFAULT_DIDO_ROOT = (SCRIPT_DIR.parent / 'DIDO').resolve()

# Prefer to IGNORE SIGPIPE (Python default), so writes to a closed pipe raise
# BrokenPipeError instead of killing the process via SIGPIPE. We'll handle
# BrokenPipeError in a safe_print helper to allow continued processing.
try:
	if hasattr(signal, 'SIGPIPE'):
		signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except Exception:
	pass


def safe_print(*args, **kwargs) -> None:
	"""Print that survives closed stdout pipes.

	If stdout is closed (e.g., due to `| head` exiting), swallow the
	BrokenPipeError, redirect future stdout to /dev/null, and continue.
	"""
	try:
		print(*args, **kwargs)
	except BrokenPipeError:
		try:
			sys.stdout = open(os.devnull, 'w')
		except Exception:
			pass


def load_list_file(path: Path) -> List[str]:
	return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith('#')]

def autodetect_lists(data_root: Path) -> Dict[str, Path]:
	mapping = {}
	for key, fn in [('train','train.txt'), ('eval','eval.txt'), ('eval','val.txt'), ('test','test.txt')]:
		p = data_root / fn
		if p.exists() and key not in mapping:  # prefer eval.txt over val.txt if both
			mapping[key] = p
	return mapping

def random_split(names: List[str], ratios: Tuple[float,float,float], seed: int) -> Dict[str,List[str]]:
	assert abs(sum(ratios) - 1.0) < 1e-6
	rng = random.Random(seed)
	shuffled = names[:]
	rng.shuffle(shuffled)
	n = len(shuffled)
	n_train = int(ratios[0]*n)
	n_val = int(ratios[1]*n)
	train = shuffled[:n_train]
	val = shuffled[n_train:n_train+n_val]
	test = shuffled[n_train+n_val:]
	return {'train': train, 'eval': val, 'test': test}


def detect_timestamp_scale(ts: np.ndarray) -> str:
	# Distinguish seconds vs nanoseconds vs something else.
	# If median <1e10 treat as seconds -> convert to ns for groundTruthPoses integer timestamp.
	med = float(np.median(ts))
	if med < 1e10:
		return 's'
	elif med < 1e13:
		return 'us'
	else:
		return 'ns'


def plot_processed_groundtruth(gt_csv: Path, out_path: Path, enable_3d: bool = True, dpi: int = 120,
							   elev: float = 35.0, azim: float = -60.0) -> bool:
	"""Create XY (and optional oblique 3D) trajectory plot using processed groundTruthPoses.csv.

	groundTruthPoses.csv format (rows): timestamp_ns, px, py, pz, qw, qx, qy, qz
	"""
	if not gt_csv.exists():
		return False
	xs = []
	ys = []
	zs = []
	with gt_csv.open('r') as f:
		for line in f:
			parts = line.strip().split(',')
			if len(parts) < 4:
				continue
			try:
				x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
			except ValueError:
				continue
			xs.append(x); ys.append(y); zs.append(z)
	if not xs:
		return False
	xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
	if enable_3d:
		fig = plt.figure(figsize=(9,4.5))
		gs = fig.add_gridspec(1,2, width_ratios=[1,1.1])
		ax2d = fig.add_subplot(gs[0,0])
		ax3d = fig.add_subplot(gs[0,1], projection='3d')
	else:
		fig, ax2d = plt.subplots(figsize=(5,5))
		ax3d = None
	# 2D view
	ax2d.plot(xs, ys, linewidth=1.0)
	ax2d.scatter([xs[0]], [ys[0]], c='green', s=30, label='start')
	ax2d.scatter([xs[-1]], [ys[-1]], c='red', s=30, label='end')
	ax2d.set_title(out_path.parent.name)
	ax2d.set_xlabel('X (m)'); ax2d.set_ylabel('Y (m)')
	ax2d.axis('equal')
	ax2d.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
	ax2d.legend(loc='best', fontsize=8)
	# 3D view
	if ax3d is not None:
		ax3d.plot3D(xs, ys, zs, linewidth=0.8)
		ax3d.scatter([xs[0]], [ys[0]], [zs[0]], c='green', s=20)
		ax3d.scatter([xs[-1]], [ys[-1]], [zs[-1]], c='red', s=20)
		ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
		ax3d.view_init(elev=elev, azim=azim)
		ax3d.set_title('Oblique 3D')
	fig.tight_layout()
	fig.savefig(out_path, dpi=dpi)
	plt.close(fig)
	return True


def convert_sequence(seq_dir: Path, out_seq_dir: Path, overwrite: bool = False,
					 plot: bool = True, plot3d: bool = True, plot_dpi: int = 120,
					 plot_name: str = 'trajectory.png', plot_elev: float = 35.0,
					 plot_azim: float = -60.0) -> None:
	h5file = seq_dir / 'data.hdf5'
	if not h5file.exists():
		safe_print(f"[WARN] Missing data.hdf5 in {seq_dir.name}, skipping")
		return
	if out_seq_dir.exists() and not overwrite:
		safe_print(f"[SKIP] {out_seq_dir} already exists")
		return
	out_seq_dir.mkdir(parents=True, exist_ok=True)

	with h5py.File(h5file, 'r') as f:
		required = ['ts', 'acc', 'gyr', 'gt_p', 'gt_q']
		for r in required:
			if r not in f:
				safe_print(f"[ERROR] Missing dataset '{r}' in {h5file}, skipping")
				return
		ts = f['ts'][:]
		acc = f['acc'][:]
		gyr = f['gyr'][:]
		gt_p = f['gt_p'][:]
		gt_q = f['gt_q'][:]
		meas_rpm = f['meas_rpm'][:] if 'meas_rpm' in f else None

	scale = detect_timestamp_scale(ts)
	if scale == 's':
		ts_seconds = ts
		ts_ns_int = (ts * 1e9).astype(np.int64)
	elif scale == 'us':
		ts_seconds = ts / 1e6
		ts_ns_int = (ts * 1e3).astype(np.int64)
	else:
		ts_seconds = ts / 1e9
		ts_ns_int = ts.astype(np.int64)

	# groundTruthPoses.csv (timestamp_ns, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
	gt_pose_path = out_seq_dir / 'groundTruthPoses.csv'
	with gt_pose_path.open('w', newline='') as fcsv:
		writer = csv.writer(fcsv)
		writer.writerow(['timestamp','p_x','p_y','p_z','q_w','q_x','q_y','q_z'])
		for t, p, q in zip(ts_ns_int, gt_p, gt_q):
			writer.writerow([int(t), *[f"{v:.6f}" for v in p], *[f"{v:.6f}" for v in q]])

	# imu_data.csv (# header, then timestamp(seconds float), acc(3), gyr(3))
	imu_path = out_seq_dir / 'imu_data.csv'
	with imu_path.open('w', newline='') as fcsv:
		writer = csv.writer(fcsv)
		writer.writerow(['timestamp','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z'])
		t0 = ts_seconds[0] if len(ts_seconds) else 0.0
		for t, a, g in zip(ts_seconds, acc, gyr):
			writer.writerow([f"{t - t0:.9f}", *[f"{v:.9f}" for v in a], *[f"{v:.9f}" for v in g]])

	# thrust_data.csv (# header, timestamp + thrust_1..thrust_4). If missing use zeros.
	thrust_path = out_seq_dir / 'thrust_data.csv'
	with thrust_path.open('w', newline='') as fcsv:
		writer = csv.writer(fcsv)
		writer.writerow(['timestamp','thrust_1','thrust_2','thrust_3','thrust_4'])
		t0 = ts_seconds[0] if len(ts_seconds) else 0.0
		if meas_rpm is not None and meas_rpm.ndim == 2:
			cols = meas_rpm.shape[1]
		else:
			cols = 0
		if cols >= 5:
			motor = meas_rpm[:,1:5]
			for t, m in zip(ts_seconds, motor):
				writer.writerow([f"{t - t0:.6f}", *[f"{v:.9f}" for v in m]])
		elif cols == 4:
			for t, m in zip(ts_seconds, meas_rpm):
				writer.writerow([f"{t - t0:.6f}", *[f"{v:.9f}" for v in m]])
		else:
			zero_row = ['0.0','0.0','0.0','0.0']
			for t in ts_seconds:
				writer.writerow([f"{t - t0:.6f}", *zero_row])

	# Plot (after CSV creation) using saved groundTruthPoses
	if plot:
		img_path = out_seq_dir / plot_name
		if not (img_path.exists() and not overwrite):
			ok = plot_processed_groundtruth(gt_pose_path, img_path, enable_3d=plot3d,
											 dpi=plot_dpi, elev=plot_elev, azim=plot_azim)
			if ok:
				pass

	safe_print(f"[OK] Wrote {out_seq_dir}")


def extract_pattern(name: str) -> str:
	parts = name.split('_')
	if 'a' in parts:
		idx = parts.index('a')
		return '_'.join(parts[:idx])
	return parts[0]

def main():
	parser = argparse.ArgumentParser(description="Convert DIDO raw to Blackbird-like format (aligned CLI with process_drd.py)")
	parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT, help='Root containing raw sequence directories (each with data.hdf5)')
	parser.add_argument('--out-dir', type=Path, default=DEFAULT_DIDO_ROOT , help='Output dataset root (absolute or relative)')
	parser.add_argument('--train-list', type=Path, help='Optional explicit train list file')
	parser.add_argument('--eval-list', type=Path, help='Optional explicit eval/val list file')
	parser.add_argument('--test-list', type=Path, help='Optional explicit test list file')
	parser.add_argument('--split-ratios', type=float, nargs=3, default=(0.7,0.15,0.15), help='Random split ratios if no lists found')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--overwrite', action='store_true')
	parser.add_argument('--plots', action='store_true', help='Generate trajectory plot images')
	parser.add_argument('--no-plot-3d', action='store_true')
	parser.add_argument('--plot-dpi', type=int, default=120)
	parser.add_argument('--plot-name', default='trajectory.png')
	parser.add_argument('--plot-elev', type=float, default=35.0)
	parser.add_argument('--plot-azim', type=float, default=-60.0)
	parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
	args = parser.parse_args()

	data_root: Path = args.data_root
	if not data_root.exists():
		safe_print(f"Data root {data_root} not found")
		return
	raw_dirs = [p for p in data_root.iterdir() if p.is_dir() and (p / 'data.hdf5').exists()]
	seq_names = [p.name for p in raw_dirs]
	if not seq_names:
		safe_print("No raw sequences found.")
		return

	# Decide splits
	lists_used = {}
	if any([args.train_list, args.eval_list, args.test_list]):
		train = load_list_file(args.train_list) if args.train_list else []
		val = load_list_file(args.eval_list) if args.eval_list else []
		test = load_list_file(args.test_list) if args.test_list else []
		lists_used['explicit'] = True
	else:
		auto = autodetect_lists(data_root)
		if {'train','eval','test'} <= set(auto.keys()):
			train = load_list_file(auto['train'])
			val = load_list_file(auto['eval'])
			test = load_list_file(auto['test'])
			lists_used['auto'] = {k:str(v) for k,v in auto.items()}
		else:
			splits = random_split(seq_names, tuple(args.split_ratios), args.seed)
			train, val, test = splits['train'], splits['eval'], splits['test']
			lists_used['random'] = args.split_ratios

	# Filter to existing
	existing = set(seq_names)
	train = [s for s in train if s in existing]
	val = [s for s in val if s in existing]
	test = [s for s in test if s in existing]
	safe_print(f"Split sizes: train={len(train)} val={len(val)} test={len(test)}")

	out_root = args.out_dir if args.out_dir.is_absolute() else args.out_dir
	out_root.mkdir(parents=True, exist_ok=True)

	work = []
	for split_name, group in [('train', train), ('eval', val), ('test', test)]:
		for s in group:
			work.append((split_name, s))

	iterator = work if args.no_progress else tqdm(work, desc='Processing DIDO sequences', unit='seq')
	for split_name, s in iterator:
		pattern = extract_pattern(s)
		out_seq = out_root / split_name / pattern / 'yawForward' / s
		convert_sequence(
			data_root / s,
			out_seq,
			overwrite=args.overwrite,
			plot=args.plots,
			plot3d=not args.no_plot_3d,
			plot_dpi=args.plot_dpi,
			plot_name=args.plot_name,
			plot_elev=args.plot_elev,
			plot_azim=args.plot_azim,
		)

	safe_print("Done.")
	if lists_used:
		safe_print("Split source:", lists_used)


if __name__ == '__main__':
	main()

