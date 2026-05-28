"""Concatenate multiple BeyondMimic NPZ motion files into one.

BeyondMimic's MotionCommand expects a single NPZ file. This script merges
multiple NPZ files (from pkl_to_npz.py) so the adaptive sampler covers all
motions in one unified timeline.

All input files must have the same fps and the same number of robot bodies.

Usage:
    python combine_npz.py --input_dir /path/to/npzs/ --output /path/to/combined.npz
"""

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of NPZ files to merge.")
    parser.add_argument("--output", type=str, required=True, help="Output combined NPZ path.")
    parser.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for input files.")
    args = parser.parse_args()

    import glob
    npz_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not npz_files:
        print(f"No NPZ files found in {args.input_dir}")
        return

    print(f"Combining {len(npz_files)} NPZ files...")

    arrays = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    fps_ref = None

    for f in npz_files:
        d = np.load(f)
        fps = int(d["fps"][0]) if d["fps"].ndim > 0 else int(d["fps"])
        if fps_ref is None:
            fps_ref = fps
        elif fps != fps_ref:
            print(f"WARNING: {f} has fps={fps}, expected {fps_ref} — skipping")
            continue
        for k in arrays:
            arrays[k].append(d[k])
        N = d["joint_pos"].shape[0]
        print(f"  {os.path.basename(f)}: {N} frames @ {fps} fps")

    combined = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    combined["fps"] = np.array([fps_ref])

    total = combined["joint_pos"].shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **combined)
    print(f"Saved combined NPZ: {total} frames @ {fps_ref} fps → {args.output}")


if __name__ == "__main__":
    main()
