"""
Sim2Sim: Load trained legged_lab AMP policy → run in MuJoCo.

Usage:
  python sim2sim_g1_amp.py --checkpoint <path/to/model_N.pt> [--cmd_vx 0.5] [--no_render]

Obs space (policy input, 585 = 117 per step × 5 history steps):
  base_ang_vel          3
  root_local_rot_tan_norm  6
  velocity_commands     3
  joint_pos             29  (absolute, alphabetical Isaac Lab order)
  joint_vel             29
  actions               29
  key_body_pos_b        18  (6 bodies × 3, in pelvis frame)
"""

import argparse
import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────── paths ───────────────────────────────────────
G1_XML = "/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml"
G1_DIR  = os.path.dirname(G1_XML)

# ──────────────────────────── joint ordering ──────────────────────────────────
# Isaac Lab uses alphabetical joint order for joint_names_expr=[".*"]
ISL_JOINTS = [
    "left_ankle_pitch_joint",   # 0
    "left_ankle_roll_joint",    # 1
    "left_elbow_joint",         # 2
    "left_hip_pitch_joint",     # 3
    "left_hip_roll_joint",      # 4
    "left_hip_yaw_joint",       # 5
    "left_knee_joint",          # 6
    "left_shoulder_pitch_joint",# 7
    "left_shoulder_roll_joint", # 8
    "left_shoulder_yaw_joint",  # 9
    "left_wrist_pitch_joint",   # 10
    "left_wrist_roll_joint",    # 11
    "left_wrist_yaw_joint",     # 12
    "right_ankle_pitch_joint",  # 13
    "right_ankle_roll_joint",   # 14
    "right_elbow_joint",        # 15
    "right_hip_pitch_joint",    # 16
    "right_hip_roll_joint",     # 17
    "right_hip_yaw_joint",      # 18
    "right_knee_joint",         # 19
    "right_shoulder_pitch_joint",# 20
    "right_shoulder_roll_joint",# 21
    "right_shoulder_yaw_joint", # 22
    "right_wrist_pitch_joint",  # 23
    "right_wrist_roll_joint",   # 24
    "right_wrist_yaw_joint",    # 25
    "waist_pitch_joint",        # 26
    "waist_roll_joint",         # 27
    "waist_yaw_joint",          # 28
]
N_JOINTS = len(ISL_JOINTS)

# Default joint positions (Isaac Lab order)
DEFAULT_POS_ISL = np.array([
    -0.2,   # left_ankle_pitch
     0.0,   # left_ankle_roll
     0.97,  # left_elbow
    -0.1,   # left_hip_pitch
     0.0,   # left_hip_roll
     0.0,   # left_hip_yaw
     0.3,   # left_knee
     0.3,   # left_shoulder_pitch
     0.25,  # left_shoulder_roll
     0.0,   # left_shoulder_yaw
     0.0,   # left_wrist_pitch
     0.15,  # left_wrist_roll
     0.0,   # left_wrist_yaw
    -0.2,   # right_ankle_pitch
     0.0,   # right_ankle_roll
     0.97,  # right_elbow
    -0.1,   # right_hip_pitch
     0.0,   # right_hip_roll
     0.0,   # right_hip_yaw
     0.3,   # right_knee
     0.3,   # right_shoulder_pitch
    -0.25,  # right_shoulder_roll
     0.0,   # right_shoulder_yaw
     0.0,   # right_wrist_pitch
    -0.15,  # right_wrist_roll
     0.0,   # right_wrist_yaw
     0.0,   # waist_pitch
     0.0,   # waist_roll
     0.0,   # waist_yaw
], dtype=np.float32)

# PD gains (Isaac Lab order) — match actuator config in legged_lab
KP_ISL = np.array([
     40, 40, 40,          # ankle_p, ankle_r, elbow (left)
    100, 100, 100, 150,   # hip_p, hip_r, hip_y, knee (left)
     40, 40, 40,          # shoulder_p, shoulder_r, shoulder_y (left)
     40, 40, 40,          # wrist_p, wrist_r, wrist_y (left)
     40, 40, 40,          # ankle_p, ankle_r, elbow (right)
    100, 100, 100, 150,   # hip_p, hip_r, hip_y, knee (right)
     40, 40, 40,          # shoulder_p, shoulder_r, shoulder_y (right)
     40, 40, 40,          # wrist_p, wrist_r, wrist_y (right)
     40, 40, 200,         # waist_pitch, waist_roll, waist_yaw
], dtype=np.float32)

KD_ISL = np.array([
     2, 2, 1,             # ankle_p, ankle_r, elbow (left)
     2, 2, 2, 4,          # hip_p, hip_r, hip_y, knee (left)
     1, 1, 1,             # shoulder (left)
     1, 1, 1,             # wrist (left)
     2, 2, 1,             # ankle_p, ankle_r, elbow (right)
     2, 2, 2, 4,          # hip_p, hip_r, hip_y, knee (right)
     1, 1, 1,             # shoulder (right)
     1, 1, 1,             # wrist (right)
     5, 5, 5,             # waist_pitch, waist_roll, waist_yaw
], dtype=np.float32)

# Effort limits (Isaac Lab order) — from actuator config in legged_lab
EFFORT_LIMIT_ISL = np.array([
     25,  25,  25,        # ankle_p, ankle_r, elbow (left)
     88, 139,  88, 139,   # hip_p, hip_r, hip_y, knee (left)
     25,  25,  25,        # shoulder (left)
      5,  25,   5,        # wrist_p, wrist_r, wrist_y (left)
     25,  25,  25,        # ankle_p, ankle_r, elbow (right)
     88, 139,  88, 139,   # hip_p, hip_r, hip_y, knee (right)
     25,  25,  25,        # shoulder (right)
      5,  25,   5,        # wrist_p, wrist_r, wrist_y (right)
     25,  25,  88,        # waist_pitch, waist_roll, waist_yaw
], dtype=np.float32)

# Key bodies for key_body_pos_b (must match KEY_BODY_NAMES in g1_amp_env_cfg.py)
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

# Action scale (from JointPositionActionCfg scale=0.25)
ACTION_SCALE = 0.25

# Obs/history config
OBS_PER_STEP  = 117  # 3+6+3+29+29+29+18
HISTORY_LEN   = 5
OBS_DIM       = OBS_PER_STEP * HISTORY_LEN  # 585


# ───────────────────────────── math helpers ──────────────────────────────────

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])


def get_yaw_quat(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from (w, x, y, z)."""
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cy, 0.0, 0.0, sy])


# ─────────────────────────── minimal actor MLP ───────────────────────────────

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256),    nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def load_actor(checkpoint_path: str) -> Actor:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
    actor = Actor(OBS_DIM, N_JOINTS)
    actor.net.load_state_dict(actor_state)
    actor.eval()
    print(f"[sim2sim] Loaded checkpoint iter={ckpt['iter']}  from {checkpoint_path}")
    return actor


# ─────────────────────────── MuJoCo setup ───────────────────────────────────

def build_model():
    """Load G1 MJCF with floor added, return (model, isl_qpos_ids, isl_dof_ids, key_body_ids, pelvis_id)."""
    # G1 MJCF already includes floor, lights, and friction — load directly
    model = mujoco.MjModel.from_xml_path(G1_XML)

    # Build qpos/dof index arrays in Isaac Lab order
    isl_qpos_ids = np.array([model.joint(n).qposadr[0] for n in ISL_JOINTS], dtype=int)
    isl_dof_ids  = np.array([model.joint(n).dofadr[0]  for n in ISL_JOINTS], dtype=int)
    # Actuator order matches MuJoCo motor definition order (same as MJC joint order).
    # We build isl→actuator mapping via joint name lookup.
    isl_act_ids  = np.array([model.actuator(n).id for n in ISL_JOINTS], dtype=int)

    key_body_ids = np.array([model.body(n).id for n in KEY_BODY_NAMES], dtype=int)
    pelvis_id    = model.body("pelvis").id

    return model, isl_qpos_ids, isl_dof_ids, isl_act_ids, key_body_ids, pelvis_id


def reset(model, data, isl_qpos_ids):
    """Reset robot to default standing pose."""
    mujoco.mj_resetData(model, data)
    data.qpos[2]  = 0.793      # height from MJCF pelvis pos
    data.qpos[3]  = 1.0        # quaternion w=1 (upright)
    data.qpos[4:7] = 0.0
    # Set joints to default position
    data.qpos[isl_qpos_ids] = DEFAULT_POS_ISL
    mujoco.mj_forward(model, data)


# ──────────────────────── observation assembly ───────────────────────────────

def get_obs_step(data, isl_qpos_ids, isl_dof_ids, key_body_ids, pelvis_id,
                 last_action: np.ndarray, cmd: np.ndarray) -> np.ndarray:
    """Compute one step of policy observations (117-dim, Isaac Lab convention)."""

    # ── base angular velocity in body frame ──────────────────────────────────
    # MuJoCo free joint qvel[3:6] = angular velocity in world frame
    ang_vel_w = data.qvel[3:6]
    R_pelvis  = data.xmat[pelvis_id].reshape(3, 3)  # world←body rotation
    ang_vel_b = R_pelvis.T @ ang_vel_w               # rotate to body frame

    # ── root_local_rot_tan_norm ──────────────────────────────────────────────
    # MuJoCo free joint stores quat as (w, x, y, z)
    root_quat     = data.qpos[3:7][[0, 1, 2, 3]]      # already (w,x,y,z)
    yaw_quat      = get_yaw_quat(root_quat)
    local_quat    = quat_mul(quat_conjugate(yaw_quat), root_quat)
    R_local       = quat_to_rotmat(local_quat)
    tan_vec       = R_local[:, 0]   # first column
    norm_vec      = R_local[:, 2]   # third column
    rot_tan_norm  = np.concatenate([tan_vec, norm_vec])  # (6,)

    # ── velocity command [vx, vy, wz] ────────────────────────────────────────
    vel_cmd = cmd.astype(np.float32)  # (3,)

    # ── joint positions (absolute, Isaac Lab order) ──────────────────────────
    joint_pos = data.qpos[isl_qpos_ids].astype(np.float32)  # (29,)

    # ── joint velocities (Isaac Lab order) ───────────────────────────────────
    joint_vel = data.qvel[isl_dof_ids].astype(np.float32)   # (29,)

    # ── last action ──────────────────────────────────────────────────────────
    action_obs = last_action.astype(np.float32)              # (29,)

    # ── key body positions in pelvis frame ───────────────────────────────────
    root_pos_w = data.xpos[pelvis_id]
    key_pos_b_list = []
    for bid in key_body_ids:
        body_pos_w = data.xpos[bid]
        pos_b = R_pelvis.T @ (body_pos_w - root_pos_w)
        key_pos_b_list.append(pos_b)
    key_body_pos_b = np.concatenate(key_pos_b_list).astype(np.float32)  # (18,)

    return np.concatenate([
        ang_vel_b,       # 3
        rot_tan_norm,    # 6
        vel_cmd,         # 3
        joint_pos,       # 29
        joint_vel,       # 29
        action_obs,      # 29
        key_body_pos_b,  # 18
    ]).astype(np.float32)  # total: 117


# ─────────────────────────────── main loop ───────────────────────────────────

def run(checkpoint: str, cmd_vx: float, cmd_vy: float, cmd_wz: float,
        max_steps: int, render: bool):

    actor = load_actor(checkpoint)

    model, isl_qpos_ids, isl_dof_ids, isl_act_ids, key_body_ids, pelvis_id = build_model()
    data = mujoco.MjData(model)

    # Command [vx, vy, wz]
    cmd = np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32)

    # History buffer: deque of obs steps
    obs_hist = deque(maxlen=HISTORY_LEN)

    last_action = np.zeros(N_JOINTS, dtype=np.float32)

    reset(model, data, isl_qpos_ids)

    # Fill history with initial obs
    init_obs = get_obs_step(data, isl_qpos_ids, isl_dof_ids, key_body_ids,
                            pelvis_id, last_action, cmd)
    for _ in range(HISTORY_LEN):
        obs_hist.append(init_obs.copy())

    # Simulation dt = 0.002 (MuJoCo), policy runs at 0.02 (10 sim steps per policy step)
    SIM_DT     = model.opt.timestep   # 0.002 set in MJCF option
    POLICY_DT  = 0.02                 # same as env step_dt (sim.dt * decimation = 0.005*4)
    SIM_PER_POLICY = max(1, int(round(POLICY_DT / SIM_DT)))  # 10

    print(f"[sim2sim] SIM_DT={SIM_DT:.4f}  POLICY_DT={POLICY_DT:.3f}  steps_per_policy={SIM_PER_POLICY}")
    print(f"[sim2sim] Command: vx={cmd_vx}  vy={cmd_vy}  wz={cmd_wz}")

    torque = np.zeros(N_JOINTS, dtype=np.float32)

    if render:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 90
    else:
        viewer = None

    step_count = 0
    policy_count = 0
    t_start = time.time()

    try:
        while step_count < max_steps:
            if step_count % SIM_PER_POLICY == 0:
                # ── assemble policy obs (history concatenated) ────────────────
                obs_vec = np.concatenate(list(obs_hist), axis=0)  # (585,)
                obs_t   = torch.from_numpy(obs_vec).unsqueeze(0)   # (1, 585)
                with torch.no_grad():
                    action = actor(obs_t).squeeze(0).numpy()       # (29,)
                last_action = action.copy()

                # ── target joint position = default + scale * action ──────────
                target_pos = DEFAULT_POS_ISL + ACTION_SCALE * action  # (29,)

                # ── PD torque ─────────────────────────────────────────────────
                cur_pos = data.qpos[isl_qpos_ids]
                cur_vel = data.qvel[isl_dof_ids]
                torque  = KP_ISL * (target_pos - cur_pos) - KD_ISL * cur_vel
                torque  = np.clip(torque, -EFFORT_LIMIT_ISL, EFFORT_LIMIT_ISL)

                # Update obs history
                obs_step = get_obs_step(data, isl_qpos_ids, isl_dof_ids,
                                        key_body_ids, pelvis_id, last_action, cmd)
                obs_hist.append(obs_step)
                policy_count += 1

            # ── apply torque via actuator order ──────────────────────────────
            data.ctrl[isl_act_ids] = torque

            mujoco.mj_step(model, data)
            step_count += 1

            if render and viewer is not None and viewer.is_running():
                viewer.sync()
            elif render and viewer is not None and not viewer.is_running():
                break

            # Check for fall or numerical blow-up
            if data.qpos[2] < 0.3 or not np.isfinite(data.qpos).all():
                print(f"[sim2sim] Robot fell at policy_step={policy_count}  t={data.time:.2f}s — resetting")
                reset(model, data, isl_qpos_ids)
                last_action = np.zeros(N_JOINTS, dtype=np.float32)
                obs_hist.clear()
                init_obs = get_obs_step(data, isl_qpos_ids, isl_dof_ids,
                                        key_body_ids, pelvis_id, last_action, cmd)
                for _ in range(HISTORY_LEN):
                    obs_hist.append(init_obs.copy())

            # Console log every 200 policy steps
            if policy_count % 200 == 0 and step_count % SIM_PER_POLICY == 0:
                height  = data.qpos[2]
                vel_x   = data.qvel[0]
                elapsed = time.time() - t_start
                print(f"  policy_step={policy_count:5d}  t={data.time:6.2f}s  "
                      f"height={height:.3f}  vel_x={vel_x:+.3f}  wall={elapsed:.1f}s")

    finally:
        if viewer is not None:
            viewer.close()

    print(f"[sim2sim] Done. {policy_count} policy steps in {time.time()-t_start:.1f}s")


# ─────────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/home/rexcon/legged_lab/scripts/rsl_rl/logs/rsl_rl/g1_amp/"
                                "2026-05-24_17-09-00/model_200.pt",
                        help="Path to model_N.pt checkpoint")
    parser.add_argument("--cmd_vx",  type=float, default=0.5,  help="Forward velocity command (m/s)")
    parser.add_argument("--cmd_vy",  type=float, default=0.0,  help="Lateral velocity command (m/s)")
    parser.add_argument("--cmd_wz",  type=float, default=0.0,  help="Yaw rate command (rad/s)")
    parser.add_argument("--steps",   type=int,   default=50000, help="Total sim steps")
    parser.add_argument("--no_render", action="store_true",    help="Disable viewer (headless)")
    args = parser.parse_args()

    run(
        checkpoint=args.checkpoint,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_wz=args.cmd_wz,
        max_steps=args.steps,
        render=not args.no_render,
    )
