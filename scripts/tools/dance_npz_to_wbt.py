"""
Convert dance_pipeline NPZ → WBT (BeyondMimic) NPZ format.

dance_pipeline NPZ keys (Pinocchio / ISL / DFS joint order):
  dof_positions   (T, 29)   ISL/DFS order
  dof_velocities  (T, 29)   ISL/DFS order
  body_positions  (T, 11, 3) 11 key bodies, body[0]=pelvis, wxyz
  body_rotations  (T, 11, 4) wxyz
  fps             scalar

WBT NPZ keys (BFS joint order, 30 robot bodies):
  joint_pos       (T, 29)   BFS order
  joint_vel       (T, 29)   BFS order
  body_pos_w      (T, 30, 3) MuJoCo body[1..30], body[0]=pelvis
  body_quat_w     (T, 30, 4) wxyz
  body_lin_vel_w  (T, 30, 3)
  body_ang_vel_w  (T, 30, 3)
  fps             (1,)

Joint reorder:  ISL/DFS → BFS  via BFS_TO_DFS permutation
Body FK:        pelvis from dance_pipeline, rest from MuJoCo mj_forward

Usage:
  python scripts/tools/dance_npz_to_wbt.py --input <dance.npz> --output <wbt.npz>
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

G1_XML = "/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml"

# ── Joint ordering permutations (from sim2sim_wbt_mujoco.py) ─────────────────

DFS_JOINT_NAMES = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",         "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",        "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",       "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

BFS_JOINT_NAMES = [
    "left_hip_pitch_joint",       "right_hip_pitch_joint",      "waist_yaw_joint",
    "left_hip_roll_joint",        "right_hip_roll_joint",       "waist_roll_joint",
    "left_hip_yaw_joint",         "right_hip_yaw_joint",        "waist_pitch_joint",
    "left_knee_joint",            "right_knee_joint",
    "left_shoulder_pitch_joint",  "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",     "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",   "right_shoulder_roll_joint",
    "left_ankle_roll_joint",      "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",    "right_shoulder_yaw_joint",
    "left_elbow_joint",           "right_elbow_joint",
    "left_wrist_roll_joint",      "right_wrist_roll_joint",
    "left_wrist_pitch_joint",     "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",       "right_wrist_yaw_joint",
]

_dfs_idx = {n: i for i, n in enumerate(DFS_JOINT_NAMES)}
# BFS_TO_DFS[j] = DFS index of BFS joint j
# joint_pos_bfs[:, j] = dof_positions_dfs[:, BFS_TO_DFS[j]]
BFS_TO_DFS = np.array([_dfs_idx[n] for n in BFS_JOINT_NAMES], dtype=int)


# ── Angular velocity helper ───────────────────────────────────────────────────

def _ang_vel_wxyz(q0, q1, dt):
    """Angular velocity (world frame) from two wxyz quaternions."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    qr_w =  w0*w1 + x0*x1 + y0*y1 + z0*z1
    qr_x = -x0*w1 + w0*x1 - z0*y1 + y0*z1
    qr_y = -y0*w1 + z0*x1 + w0*y1 - x0*z1
    qr_z = -z0*w1 - y0*x1 + x0*y1 + w0*z1
    qr_w = float(np.clip(qr_w, -1.0, 1.0))
    angle = 2.0 * np.arccos(abs(qr_w))
    sin_h = np.sqrt(max(1.0 - qr_w**2, 1e-12))
    if sin_h < 1e-6:
        return np.zeros(3, np.float32)
    axis = np.array([qr_x, qr_y, qr_z]) / sin_h
    if qr_w < 0:
        angle = -angle
    return (angle / dt * axis).astype(np.float32)


# ── Main conversion ───────────────────────────────────────────────────────────

def convert(input_path: str, output_path: str) -> None:
    src = np.load(input_path, allow_pickle=True)

    dof_pos_dfs = src["dof_positions"].astype(np.float32)   # (T, 29) ISL/DFS
    dof_vel_dfs = src["dof_velocities"].astype(np.float32)  # (T, 29)
    pelvis_pos  = src["body_positions"][:, 0, :].astype(np.float32)  # (T, 3)
    pelvis_quat = src["body_rotations"][:, 0, :].astype(np.float32)  # (T, 4) wxyz
    fps         = float(src["fps"])
    dt          = 1.0 / fps
    T           = dof_pos_dfs.shape[0]

    # Verify dof_names match DFS order
    src_names = list(src["dof_names"])
    if src_names != DFS_JOINT_NAMES:
        # Build a reorder in case they differ
        src_idx = {n: i for i, n in enumerate(src_names)}
        reorder = np.array([src_idx[n] for n in DFS_JOINT_NAMES], dtype=int)
        dof_pos_dfs = dof_pos_dfs[:, reorder]
        dof_vel_dfs = dof_vel_dfs[:, reorder]
        print("[convert] Applied dof_names reorder to match DFS order")

    # 1. Reorder joints: DFS → BFS
    joint_pos = dof_pos_dfs[:, BFS_TO_DFS]   # (T, 29) BFS
    joint_vel = dof_vel_dfs[:, BFS_TO_DFS]   # (T, 29) BFS

    # 2. FK: compute all 30 robot body poses via MuJoCo
    #    MuJoCo body[0]=world, body[1..30]=robot → NPZ body[0..29]
    mj_model = mujoco.MjModel.from_xml_path(G1_XML)
    mj_data  = mujoco.MjData(mj_model)
    N_bodies = mj_model.nbody - 1   # exclude world body (should be 30)
    assert N_bodies == 30, f"Expected 30 robot bodies, got {N_bodies}"

    body_pos_w  = np.zeros((T, N_bodies, 3),  dtype=np.float32)
    body_quat_w = np.zeros((T, N_bodies, 4),  dtype=np.float32)

    for t in range(T):
        mj_data.qpos[0:3] = pelvis_pos[t]
        mj_data.qpos[3:7] = pelvis_quat[t]        # wxyz, MuJoCo free-joint is wxyz
        mj_data.qpos[7:36] = dof_pos_dfs[t]       # DFS order matches qpos[7:36]
        mujoco.mj_forward(mj_model, mj_data)
        body_pos_w[t]  = mj_data.xpos[1:31]       # bodies 1..30
        body_quat_w[t] = mj_data.xquat[1:31]      # wxyz

    # 3. Body linear velocities (central differences, smoothed)
    body_lin_vel = np.zeros_like(body_pos_w)
    body_lin_vel[1:-1] = (body_pos_w[2:] - body_pos_w[:-2]) / (2 * dt)
    body_lin_vel[0]    = (body_pos_w[1]  - body_pos_w[0])   / dt
    body_lin_vel[-1]   = (body_pos_w[-1] - body_pos_w[-2])  / dt
    body_lin_vel = gaussian_filter1d(body_lin_vel, sigma=1, axis=0).astype(np.float32)

    # 4. Body angular velocities
    body_ang_vel = np.zeros_like(body_pos_w)
    for b in range(N_bodies):
        quats = body_quat_w[:, b, :]
        av = np.zeros((T, 3), np.float32)
        for t in range(1, T - 1):
            av[t] = _ang_vel_wxyz(quats[t - 1], quats[t + 1], 2 * dt)
        av[0]  = _ang_vel_wxyz(quats[0],  quats[1],  dt)
        av[-1] = _ang_vel_wxyz(quats[-2], quats[-1], dt)
        body_ang_vel[:, b] = gaussian_filter1d(av, sigma=1, axis=0)
    body_ang_vel = body_ang_vel.astype(np.float32)

    # 5. Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        joint_pos      = joint_pos,
        joint_vel      = joint_vel,
        body_pos_w     = body_pos_w,
        body_quat_w    = body_quat_w,
        body_lin_vel_w = body_lin_vel,
        body_ang_vel_w = body_ang_vel,
        fps            = np.array([fps], dtype=np.float32),
    )
    print(f"[convert] {T} frames @ {fps:.1f} fps → {output_path}")
    print(f"          joint_pos {joint_pos.shape}  body_pos_w {body_pos_w.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="dance_pipeline NPZ")
    parser.add_argument("--output", required=True, help="WBT NPZ output path")
    args = parser.parse_args()
    convert(args.input, args.output)
