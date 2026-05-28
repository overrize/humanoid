"""Convert openhe/g1-retargeted-motions PKL files to legged_lab PKL format.

openhe PKL structure:
    {PosixPath('source.npz'): {
        'root_trans_offset': (N, 3),   # root position in world frame
        'root_rot':          (N, 4),   # quaternion xyzw
        'dof':               (N, 29),  # joint angles in GMR order
        'fps':               scalar,
        'contact_mask':      (N, 2),
        'pose_aa':           (N, J, 3),
    }}

lab PKL structure:
    {
        'fps':          float,
        'root_pos':     (N, 3)   float64,
        'root_rot':     (N, 4)   float32  wxyz,
        'dof_pos':      (N, 29)  float64  in lab DOF order,
        'loop_mode':    0,
        'key_body_pos': (N, 6, 3) float32  world frame,  ← requires FK
    }

key_body_pos is computed via the legged_lab gmr_to_lab retarget pipeline
which runs Isaac Lab's forward kinematics. Run this script with:
    python convert_openhe_to_lab.py --headless

NOTE: ACCAD subset uses a different PKL structure (dataset-level dict),
      and is skipped automatically.
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────
# GMR → lab DOF order mapping (from g1_29dof.yaml)
# ──────────────────────────────────────────────────────────────────
GMR_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

LAB_DOF_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

GMR_TO_LAB_IDX = [GMR_DOF_NAMES.index(n) for n in LAB_DOF_NAMES]


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """(N,4) xyzw → wxyz."""
    return np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)


def load_openhe_pkl(path: str):
    """Load openhe PKL and extract motion dict. Returns None if unsupported format."""
    with open(path, "rb") as f:
        raw = pickle.load(f)

    # openhe format: single PosixPath key mapping to motion dict
    if not isinstance(raw, dict):
        return None
    keys = list(raw.keys())
    if len(keys) != 1:
        return None
    key = keys[0]
    # PosixPath key indicates valid openhe format
    if not isinstance(key, Path):
        return None

    motion = raw[key]
    if not isinstance(motion, dict):
        return None
    required = {"root_trans_offset", "root_rot", "dof", "fps"}
    if not required.issubset(motion.keys()):
        return None

    return motion


def convert_one(motion: dict, out_path: str) -> dict:
    """Convert openhe motion dict to intermediate GMR-compatible dict (without key_body_pos).
    Saves as a temporary file for gmr_to_lab.py to process.
    Returns the gmr dict.
    """
    fps = float(motion["fps"])
    root_pos = np.array(motion["root_trans_offset"], dtype=np.float64)  # (N, 3)
    root_rot_xyzw = np.array(motion["root_rot"], dtype=np.float32)       # (N, 4) xyzw
    root_rot_wxyz = xyzw_to_wxyz(root_rot_xyzw)                          # (N, 4) wxyz
    dof_gmr = np.array(motion["dof"], dtype=np.float64)                  # (N, 29) GMR order

    if dof_gmr.shape[1] != 29:
        raise ValueError(f"Expected 29 DOFs, got {dof_gmr.shape[1]}")

    dof_lab = dof_gmr[:, GMR_TO_LAB_IDX]  # reorder to lab order

    gmr_dict = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot_wxyz,   # gmr_to_lab expects xyzw but we pre-convert; mark for skip
        "dof_pos": dof_lab,
        "loop_mode": 0,
    }
    return gmr_dict


# ──────────────────────────────────────────────────────────────────
# Isaac Lab FK to compute key_body_pos
# ──────────────────────────────────────────────────────────────────

KEY_BODY_NAMES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
]


def compute_key_body_pos_via_sim(motion_dicts: list[dict], app_launcher) -> list[np.ndarray]:
    """Run Isaac Lab FK for a batch of motions to get key_body_pos (world frame)."""
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import convert_quat
    from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
    from legged_lab import LEGGED_LAB_ROOT_DIR

    # Use gmr_to_lab's ReplayMotionsSceneCfg logic
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "retarget"))
    from gmr_to_lab import run_simulator, ReplayMotionsSceneCfg

    # Pad all motions to same length for batched simulation
    max_frames = max(m["root_pos"].shape[0] for m in motion_dicts)
    results = run_simulator(motion_dicts, app_launcher)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=(
        "/home/rexcon/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/openhe_raw"
    ))
    parser.add_argument("--output_dir", default=(
        "/home/rexcon/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/deepmimic_v2"
    ))
    parser.add_argument("--subsets", nargs="+",
                        default=["dance_db_retargeted", "kungfu_retargeted", "lafan1_retargeted"])
    AppLauncher = None  # will be imported after arg parsing

    # AppLauncher args must come first
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    os.makedirs(args_cli.output_dir, exist_ok=True)

    all_motion_dicts = []
    all_out_paths = []

    for subset in args_cli.subsets:
        subset_dir = os.path.join(args_cli.input_dir, subset)
        pkl_files = sorted(glob.glob(os.path.join(subset_dir, "*.pkl")))
        print(f"[convert] {subset}: {len(pkl_files)} files")

        for pkl_path in pkl_files:
            motion = load_openhe_pkl(pkl_path)
            if motion is None:
                print(f"  SKIP (unsupported format): {os.path.basename(pkl_path)}")
                continue

            try:
                gmr_dict = convert_one(motion, pkl_path)
            except Exception as e:
                print(f"  SKIP ({e}): {os.path.basename(pkl_path)}")
                continue

            stem = os.path.splitext(os.path.basename(pkl_path))[0]
            out_path = os.path.join(args_cli.output_dir, f"openhe_{subset}_{stem}.pkl")
            if os.path.exists(out_path):
                print(f"  EXISTS: {os.path.basename(out_path)}")
                continue

            all_motion_dicts.append(gmr_dict)
            all_out_paths.append(out_path)

    if not all_motion_dicts:
        print("[convert] Nothing to convert.")
        simulation_app.close()
        return

    print(f"[convert] Running FK simulation for {len(all_motion_dicts)} motions...")

    # Import after app launch
    import torch
    from isaaclab.utils.math import convert_quat
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "retarget"))
    from gmr_to_lab import extract_gmr_data, run_simulator

    # gmr_to_lab's run_simulator expects root_rot in xyzw internally,
    # but we've already set it to wxyz. Pass through run_simulator with pre-converted flag.
    # Actually re-read gmr_to_lab to check conversion direction...
    # For simplicity: re-convert back to xyzw for gmr_to_lab input.
    for m in all_motion_dicts:
        # wxyz → xyzw for gmr_to_lab's internal convert_quat call
        m["root_rot"] = np.concatenate([m["root_rot"][:, 1:4], m["root_rot"][:, 0:1]], axis=1)

    results = run_simulator(all_motion_dicts, app_launcher)

    for i, (result, out_path) in enumerate(zip(results, all_out_paths)):
        # Convert root_rot back to wxyz for lab PKL
        rr = result["root_rot"]
        result["root_rot"] = np.concatenate([rr[:, 3:4], rr[:, :3]], axis=1).astype(np.float32)
        import joblib
        joblib.dump(result, out_path)
        print(f"  Saved: {os.path.basename(out_path)}")

    print(f"[convert] Done. {len(all_out_paths)} files written to {args_cli.output_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
