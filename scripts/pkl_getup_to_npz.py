"""Convert AMP get-up PKL clips to BeyondMimic NPZ format.

PKL fields (post-retargeting, Isaac Lab convention):
  fps      : int (30)
  root_pos : (T, 3)   pelvis world position
  root_rot : (T, 4)   pelvis quaternion  wxyz
  dof_pos  : (T, 29)  joint angles in GMR/DFS order

Output NPZ fields:
  fps            (1,)
  joint_pos      (T, 29)  BFS order
  joint_vel      (T, 29)  BFS order
  body_pos_w     (T, 30, 3)
  body_quat_w    (T, 30, 4) wxyz
  body_lin_vel_w (T, 30, 3)
  body_ang_vel_w (T, 30, 3)

Usage:
    conda run -n env_isaaclab python scripts/pkl_getup_to_npz.py \
        --output /tmp/getup_mocap_g1_50fps.npz
"""

import argparse
import os

import joblib
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
XML = "/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml"
DATA_DIR = "/tmp/gitee_legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/get_up"

# Clips with weight=1.0 from g1_amp_get_up_env_cfg.py
CLIPS = [
    "fallAndGetUp1_subject1_1060_1150",
    "fallAndGetUp1_subject1_1400_1480",
    "fallAndGetUp1_subject1_2100_2200",
    "fallAndGetUp1_subject5_2500_2600",
    "fallAndGetUp2_subject2_850_1050",
    "fallAndGetUp2_subject3_900_1000",
    "fallAndGetUp6_subject1_530_600",
    "fallAndGetUp6_subject1_650_700",
    "fallAndGetUp6_subject1_1080_1180",
    "fallAndGetUp6_subject1_1630_1690",
]

TARGET_FPS = 50

# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------
# GMR/DFS order (how dof_pos is stored in PKL)
DFS_NAMES = [
    "left_hip_pitch_joint",     "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",          "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",    "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",         "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",          "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",         "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",        "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Isaac Lab BFS order (how joint_pos is stored in NPZ)
BFS_NAMES = [
    "left_hip_pitch_joint",     "right_hip_pitch_joint",  "waist_yaw_joint",
    "left_hip_roll_joint",      "right_hip_roll_joint",   "waist_roll_joint",
    "left_hip_yaw_joint",       "right_hip_yaw_joint",    "waist_pitch_joint",
    "left_knee_joint",          "right_knee_joint",
    "left_shoulder_pitch_joint","right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",   "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint",    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",  "right_shoulder_yaw_joint",
    "left_elbow_joint",         "right_elbow_joint",
    "left_wrist_roll_joint",    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",   "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",     "right_wrist_yaw_joint",
]

_dfs_idx = {n: i for i, n in enumerate(DFS_NAMES)}
# BFS_TO_DFS[i] = index in DFS that gives BFS joint i
BFS_TO_DFS = np.array([_dfs_idx[n] for n in BFS_NAMES], dtype=int)

# ---------------------------------------------------------------------------
# BFS body ordering for FK extraction
# ---------------------------------------------------------------------------
# Maps NPZ body index (0..29) → MuJoCo body id (1..30)
BFS_MJ = np.array(
    [1, 2, 8,14,  3, 9,15,  4,10,16,  5,11,17,24,  6,12,18,25,  7,13,19,26,
     20,27, 21,28, 22,29, 23,30],
    dtype=int,
)

# ---------------------------------------------------------------------------
# Helpers (same as generate_getup_npz.py)
# ---------------------------------------------------------------------------
def _extract_bodies(data):
    bp = np.array([data.xpos[b]  for b in BFS_MJ], dtype=np.float32)   # (30,3)
    bq = np.array([data.xquat[b] for b in BFS_MJ], dtype=np.float32)   # (30,4) wxyz
    return bp, bq


def _ang_vel_from_quats(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Body angular velocity (rad/s, world frame) from quaternion sequence."""
    T = quats_wxyz.shape[0]
    ang_vel = np.zeros((T, 3), dtype=np.float32)
    for i in range(T - 1):
        q0 = quats_wxyz[i]
        q1 = quats_wxyz[i + 1]
        # dq/dt ≈ (q1 - q0) / dt
        dq = (q1 - q0) / dt
        # ω = 2 * q_inv ⊗ dq_dt   (in body frame) → convert to world
        w, x, y, z = q0
        qinv = np.array([w, -x, -y, -z])
        qv   = dq
        # quaternion multiply qinv ⊗ qv → [0, ω_body]
        om_body = 2.0 * np.array([
            qinv[0]*qv[1] + qinv[1]*qv[0] + qinv[2]*qv[3] - qinv[3]*qv[2],
            qinv[0]*qv[2] - qinv[1]*qv[3] + qinv[2]*qv[0] + qinv[3]*qv[1],
            qinv[0]*qv[3] + qinv[1]*qv[2] - qinv[2]*qv[1] + qinv[3]*qv[0],
        ], dtype=np.float32)
        # Rotate to world frame: ω_world = q ⊗ ω_body ⊗ q_inv
        R = Rotation.from_quat([x, y, z, w]).as_matrix()
        ang_vel[i] = R @ om_body
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def _process_clip(pkl_path: str, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Load one PKL clip, resample to TARGET_FPS, run FK, return arrays."""
    d     = joblib.load(pkl_path)
    src_fps   = int(d["fps"])
    root_pos  = d["root_pos"].astype(np.float64)   # (T, 3)
    root_rot  = d["root_rot"].astype(np.float64)   # (T, 4) wxyz
    dof_dfs   = d["dof_pos"].astype(np.float64)    # (T, 29) DFS

    # Clamp to physical joint limits to avoid out-of-range MoCap retargeting angles
    for i, name in enumerate(DFS_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        orig_min, orig_max = dof_dfs[:, i].min(), dof_dfs[:, i].max()
        dof_dfs[:, i] = np.clip(dof_dfs[:, i], lo, hi)
        if orig_min < lo - 0.01 or orig_max > hi + 0.01:
            print(f"    [clamp] {name}: [{orig_min:.3f}, {orig_max:.3f}] → [{lo:.3f}, {hi:.3f}]")

    T_src = root_pos.shape[0]
    t_src = np.arange(T_src) / src_fps
    dur   = t_src[-1]

    T_tgt = int(round(dur * TARGET_FPS)) + 1
    t_tgt = np.arange(T_tgt) / TARGET_FPS
    t_tgt = np.clip(t_tgt, 0.0, dur)

    # Resample root_pos (cubic)
    cs_rp  = CubicSpline(t_src, root_pos)
    rp_r   = cs_rp(t_tgt).astype(np.float32)   # (T_tgt, 3)

    # Resample root_rot (SLERP)
    rot_r_scipy = Rotation.from_quat(root_rot[:, [1, 2, 3, 0]])  # wxyz→xyzw
    slerp       = Slerp(t_src, rot_r_scipy)
    rr_xyzw     = slerp(t_tgt).as_quat()                         # (T_tgt, 4) xyzw
    rr_wxyz     = rr_xyzw[:, [3, 0, 1, 2]].astype(np.float32)   # →wxyz

    # Resample dof_pos (cubic)
    cs_dof = CubicSpline(t_src, dof_dfs)
    dof_r  = cs_dof(t_tgt).astype(np.float32)   # (T_tgt, 29) DFS
    dvel_r = cs_dof(t_tgt, 1).astype(np.float32)

    # DFS → BFS reorder
    jp_bfs  = dof_r[:,  BFS_TO_DFS]   # (T_tgt, 29) BFS
    jv_bfs  = dvel_r[:, BFS_TO_DFS]

    # Ground-height normalisation: the retargeting places the robot so the
    # lowest pelvis height in the clip is ~on the ground.  We apply a single
    # z-offset so the minimum root_pos z across the clip is at SUPINE_PELVIS_Z
    # (the expected pelvis height when lying flat), leaving the trajectory's
    # internal structure intact.
    # Extreme joint angles can place sensor spheres underground, so we avoid
    # per-frame geom scanning and use root_pos z as the reference instead.
    SUPINE_PELVIS_Z = 0.12   # expected pelvis z when robot is supine (metres)
    min_rp_z = float(rp_r[:, 2].min())
    z_offset  = SUPINE_PELVIS_Z - min_rp_z
    rp_r[:, 2] += z_offset

    # Run FK for all frames
    T = T_tgt
    body_pos  = np.zeros((T, 30, 3), np.float32)
    body_quat = np.zeros((T, 30, 4), np.float32)
    for f in range(T):
        data.qpos[0:3]  = rp_r[f]
        data.qpos[3:7]  = rr_wxyz[f]
        data.qpos[7:36] = dof_r[f]
        data.qvel[:]    = 0.0
        mujoco.mj_forward(model, data)
        body_pos[f], body_quat[f] = _extract_bodies(data)

    dt = 1.0 / TARGET_FPS
    body_lin_vel = np.zeros_like(body_pos)
    body_lin_vel[:-1] = (body_pos[1:] - body_pos[:-1]) / dt
    body_lin_vel[-1]  = body_lin_vel[-2]

    body_ang_vel = np.zeros_like(body_pos)
    for b in range(30):
        body_ang_vel[:, b] = _ang_vel_from_quats(body_quat[:, b], dt)

    return dict(
        joint_pos      = jp_bfs,
        joint_vel      = jv_bfs,
        body_pos_w     = body_pos,
        body_quat_w    = body_quat,
        body_lin_vel_w = body_lin_vel,
        body_ang_vel_w = body_ang_vel,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/getup_mocap_g1_50fps.npz")
    parser.add_argument("--data_dir", default=DATA_DIR)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    data  = mujoco.MjData(model)

    all_clips = []
    for clip_name in CLIPS:
        pkl_path = os.path.join(args.data_dir, f"{clip_name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  SKIP (not found): {clip_name}")
            continue
        print(f"  Processing {clip_name} …", end=" ", flush=True)
        arrays = _process_clip(pkl_path, model, data)
        T = arrays["joint_pos"].shape[0]
        z0 = arrays["body_pos_w"][0, 0, 2]
        z1 = arrays["body_pos_w"][-1, 0, 2]
        print(f"T={T}  pelvis_z: {z0:.3f} → {z1:.3f}")
        all_clips.append(arrays)

    if not all_clips:
        print("No clips found – check DATA_DIR")
        return

    # Concatenate all clips
    merged = {k: np.concatenate([c[k] for c in all_clips], axis=0) for k in all_clips[0]}
    T_total = merged["joint_pos"].shape[0]

    np.savez(
        args.output,
        fps            = np.array([TARGET_FPS], dtype=np.int64),
        **merged,
    )
    print(f"\nSaved → {args.output}")
    print(f"  {len(all_clips)} clips  {T_total} frames total  ({T_total/TARGET_FPS:.1f}s @ {TARGET_FPS}fps)")
    print(f"  joint_pos:  {merged['joint_pos'].shape}")
    print(f"  body_pos_w: {merged['body_pos_w'].shape}")


if __name__ == "__main__":
    main()
