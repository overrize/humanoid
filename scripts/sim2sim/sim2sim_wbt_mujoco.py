"""MuJoCo sim2sim for BeyondMimic WBT G1 policy.

Key design notes:
  1. Joint ordering: Isaac Lab stores joints in BFS order; MuJoCo uses DFS order.
     BFS↔DFS permutation arrays (bfs_to_dfs / dfs_to_bfs) handle all conversions.
  2. RefFK: a second MuJoCo model instance computes reference body FK from the same
     model as the simulation, so relative anchor errors are consistent regardless of
     absolute kinematic differences between the BeyondMimic URDF and g1_29dof.xml.
  3. Height correction: the NPZ pelvis z was recorded in Isaac Lab (different URDF);
     we snap the robot to ground and apply the same offset to the reference FK.
  4. PD control: armature + implicit joint damping are set in the model (matching
     training values). Only the stiffness term is applied explicitly as torque.
  5. Starting frame: playback auto-selects the NPZ frame closest to DEFAULT_BFS so
     the policy starts well within its training distribution (avoids stiff-leg symptom
     that occurs when frame 0 has nearly-straight knees ≈ 8σ from training default).

Robustness testing (keyboard):
  Arrow keys  — apply 200 N push in forward/back/left/right direction
  P           — apply 200 N push in a random horizontal direction
  --push_force N   — change push magnitude (default 200 N)
  --push_duration S — change push duration (default 0.08 s)
  --push_interval S — auto-apply random push every S seconds (0 = disabled)

Mouse perturbation: double-click a body in the viewer to select it, then
  Ctrl + left-drag to pull it with a spring force.

Usage (from repo root):
    conda run -n env_isaaclab python scripts/sim2sim/sim2sim_wbt_mujoco.py \\
        --checkpoint models/wbt_g1_v1/model_29999.pt \\
        --motion_file /tmp/combined_g1_50fps.npz

    # Standing balance test
    ... --standing

    # Follow the reference motion (auto-selects best start frame)
    ... --playback

    # Follow the reference motion starting from a specific frame
    ... --playback --frame 4650

    # Playback with auto random push every 5 seconds
    ... --playback --push_interval 5
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Body index constants (MuJoCo DFS)
# ---------------------------------------------------------------------------
MJ_PELVIS_ID = 1   # pelvis (floating base)
MJ_ANCHOR_ID = 16  # torso_link

# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------
# MuJoCo qpos[7:36] follows DFS traversal of the kinematic tree:
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

# Isaac Lab / NPZ joint_pos uses BFS traversal of the same tree:
BFS_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8
    "left_knee_joint",            # 9
    "right_knee_joint",           # 10
    "left_shoulder_pitch_joint",  # 11
    "right_shoulder_pitch_joint", # 12
    "left_ankle_pitch_joint",     # 13
    "right_ankle_pitch_joint",    # 14
    "left_shoulder_roll_joint",   # 15
    "right_shoulder_roll_joint",  # 16
    "left_ankle_roll_joint",      # 17
    "right_ankle_roll_joint",     # 18
    "left_shoulder_yaw_joint",    # 19
    "right_shoulder_yaw_joint",   # 20
    "left_elbow_joint",           # 21
    "right_elbow_joint",          # 22
    "left_wrist_roll_joint",      # 23
    "right_wrist_roll_joint",     # 24
    "left_wrist_pitch_joint",     # 25
    "right_wrist_pitch_joint",    # 26
    "left_wrist_yaw_joint",       # 27
    "right_wrist_yaw_joint",      # 28
]

# bfs_to_dfs[j] = DFS index of BFS joint j
# i.e., q_bfs[j] = q_dfs[bfs_to_dfs[j]]
_dfs_name_to_idx = {n: i for i, n in enumerate(DFS_JOINT_NAMES)}
BFS_TO_DFS = np.array([_dfs_name_to_idx[n] for n in BFS_JOINT_NAMES], dtype=int)

# dfs_to_bfs[i] = BFS index of DFS joint i
# i.e., action_dfs[i] = action_bfs[dfs_to_bfs[i]]
DFS_TO_BFS = np.zeros(29, dtype=int)
for j, i in enumerate(BFS_TO_DFS):
    DFS_TO_BFS[i] = j

# ---------------------------------------------------------------------------
# Default joint positions
# ---------------------------------------------------------------------------
_JP_DEFAULT = {
    "left_hip_pitch_joint":  -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint":        0.669, "right_knee_joint":       0.669,
    "left_ankle_pitch_joint":-0.363, "right_ankle_pitch_joint":-0.363,
    "left_elbow_joint":       0.600, "right_elbow_joint":      0.600,
    "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint":-0.2,"right_shoulder_pitch_joint": 0.2,
}
# In DFS order (matches MuJoCo qpos[7:36])
DEFAULT_DFS = np.array([_JP_DEFAULT.get(j, 0.0) for j in DFS_JOINT_NAMES], np.float32)
# In BFS order (matches NPZ, policy obs/action)
DEFAULT_BFS = DEFAULT_DFS[BFS_TO_DFS]


# ---------------------------------------------------------------------------
# PD parameters (BeyondMimic G1 config, indexed in DFS order for MuJoCo)
# ---------------------------------------------------------------------------

def _make_pd_params():
    """Return (action_scale_bfs, kp_dfs, kd_dfs, tau_max_dfs, armature_dfs)."""
    OMEGA = 10.0 * 2.0 * math.pi
    DR    = 2.0
    A5020    = 0.003609725
    A7520_14 = 0.010177520
    A7520_22 = 0.025101925
    A4010    = 0.00425

    def _arm(j):
        if "hip_pitch" in j or "hip_yaw" in j or "waist_yaw" in j:  return A7520_14
        if "hip_roll"  in j or "knee"    in j:                       return A7520_22
        if "ankle"     in j or "waist_roll" in j or "waist_pitch" in j: return 2*A5020
        if "wrist_pitch" in j or "wrist_yaw" in j:                   return A4010
        return A5020

    def _tau(j):
        if "hip_pitch" in j or "hip_yaw" in j or "waist_yaw" in j:  return 88.0
        if "hip_roll"  in j or "knee"    in j:                       return 139.0
        if "ankle"     in j or "waist_roll" in j or "waist_pitch" in j: return 50.0
        if "wrist_pitch" in j or "wrist_yaw" in j:                   return 5.0
        return 25.0

    arm_dfs = np.array([_arm(j) for j in DFS_JOINT_NAMES], np.float32)
    kp_dfs  = arm_dfs * OMEGA**2
    kd_dfs  = 2 * DR * arm_dfs * OMEGA
    tau_dfs = np.array([_tau(j) for j in DFS_JOINT_NAMES], np.float32)

    # action_scale in BFS order (matches policy output)
    arm_bfs = arm_dfs[BFS_TO_DFS]
    tau_bfs = tau_dfs[BFS_TO_DFS]
    kp_bfs  = kp_dfs[BFS_TO_DFS]
    action_scale_bfs = 0.25 * tau_bfs / kp_bfs

    return action_scale_bfs, kp_dfs, kd_dfs, tau_dfs, arm_dfs


# ---------------------------------------------------------------------------
# Math (quaternion wxyz)
# ---------------------------------------------------------------------------

def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], np.float64)

def _qmul(a, b):
    w1,x1,y1,z1 = a;  w2,x2,y2,z2 = b
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2,
    ], np.float64)

def _qrot(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _qmul(_qmul(q, qv), _qinv(q))[1:]

def _q2R(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),  2*(x*z+w*y)],
        [  2*(x*y+w*z),1-2*(x*x+z*z),  2*(y*z-w*x)],
        [  2*(x*z-w*y),  2*(y*z+w*x),1-2*(x*x+y*y)],
    ], np.float64)

def _subtract_frames(t01, q01, t02, q02):
    qi = _qinv(q01)
    return _qrot(qi, t02 - t01), _qmul(qi, q02)


# ---------------------------------------------------------------------------
# Height utilities
# ---------------------------------------------------------------------------

def _lowest_geom_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    low = np.inf
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0:
            continue
        gt  = model.geom_type[i]
        pos = data.geom_xpos[i]
        sz  = model.geom_size[i]
        mat = data.geom_xmat[i].reshape(3, 3)
        if gt == 2:    # sphere
            bottom = pos[2] - sz[0]
        elif gt == 7:  # capsule — axis is local z (3rd column of mat)
            axis_z = abs(mat[2, 2])
            bottom = pos[2] - sz[0] - sz[1] * axis_z
        else:
            bottom = pos[2] - max(sz[0], sz[1] if sz.size > 1 else sz[0])
        if bottom < low:
            low = bottom
    return low if low < np.inf else 0.0


def _snap_to_ground(model: mujoco.MjModel, data: mujoco.MjData,
                    margin: float = 0.001) -> float:
    """Adjust qpos[2] so the lowest robot geom is at z=margin. Returns correction."""
    mujoco.mj_forward(model, data)
    low = _lowest_geom_z(model, data)
    corr = margin - low
    data.qpos[2] += corr
    mujoco.mj_forward(model, data)
    return corr


# ---------------------------------------------------------------------------
# Reference FK helper
# ---------------------------------------------------------------------------

class RefFK:
    """Compute reference body poses with a dedicated MuJoCo model instance.

    NPZ joint_pos is in BFS order; converted to DFS before setting qpos.
    A height_offset corrects for the kinematic difference between the
    BeyondMimic URDF (used in training) and g1_29dof.xml (used here).
    """

    def __init__(self, xml_path: str, npz_jp_bfs: np.ndarray,
                 npz_bpos: np.ndarray, npz_bquat: np.ndarray,
                 height_offset: float = 0.0):
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data  = mujoco.MjData(self._model)
        self._bpos  = npz_bpos    # (T, 30, 3) BFS body order, NPZ BFS body 0=pelvis
        self._bquat = npz_bquat   # (T, 30, 4) wxyz
        self._jp    = npz_jp_bfs  # (T, 29) BFS joint order
        self._h     = height_offset

    def get(self, frame: int):
        t = frame % self._jp.shape[0]
        self._data.qpos[0:3]  = self._bpos[t, 0]
        self._data.qpos[2]   += self._h
        self._data.qpos[3:7]  = self._bquat[t, 0]               # wxyz
        self._data.qpos[7:36] = self._jp[t][DFS_TO_BFS]         # BFS→DFS
        mujoco.mj_forward(self._model, self._data)
        return (
            self._data.xpos[MJ_ANCHOR_ID].copy(),
            self._data.xquat[MJ_ANCHOR_ID].copy(),
        )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def _build_actor(checkpoint_path: str) -> nn.Module:
    sd  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    msd = sd["model_state_dict"]
    actor = nn.Sequential(
        nn.Linear(160, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 29),
    )
    for wi in [0, 2, 4, 6]:
        actor[wi].weight.data = msd[f"actor.{wi}.weight"]
        actor[wi].bias.data   = msd[f"actor.{wi}.bias"]
    actor.eval()
    print(f"[sim2sim] Actor loaded from {checkpoint_path}")
    return actor


# ---------------------------------------------------------------------------
# Observation (160-dim, all joint quantities in BFS order to match training)
# ---------------------------------------------------------------------------

def _build_obs(data: mujoco.MjData,
               ref_jp_bfs: np.ndarray,
               ref_jv_bfs: np.ndarray,
               ref_torso_pos: np.ndarray,
               ref_torso_quat: np.ndarray,
               last_action_bfs: np.ndarray) -> np.ndarray:
    """Build 160-dim observation matching BeyondMimic PolicyCfg layout.

    [0:29]    command: ref joint positions (BFS)
    [29:58]   command: ref joint velocities (BFS)
    [58:61]   motion_anchor_pos_b
    [61:67]   motion_anchor_ori_b  (6D rotation)
    [67:70]   base_lin_vel  (pelvis body frame)
    [70:73]   base_ang_vel  (pelvis body frame)
    [73:102]  joint_pos_rel = q_bfs - default_bfs
    [102:131] joint_vel (BFS)
    [131:160] last_action (BFS)
    """
    # qvel[0:3]: linear velocity in world frame → rotate to body frame
    # qvel[3:6]: angular velocity already in body frame (MuJoCo free-joint convention)
    R_p = data.xmat[MJ_PELVIS_ID].reshape(3, 3)
    base_lin_vel = (R_p.T @ data.qvel[0:3]).astype(np.float32)
    base_ang_vel = data.qvel[3:6].astype(np.float32)

    # Joint quantities: MuJoCo gives DFS; convert to BFS for policy
    q_dfs  = data.qpos[7:36]
    dq_dfs = data.qvel[6:35]
    q_bfs  = q_dfs[BFS_TO_DFS].astype(np.float32)
    dq_bfs = dq_dfs[BFS_TO_DFS].astype(np.float32)
    joint_pos_rel_bfs = q_bfs - DEFAULT_BFS
    joint_vel_bfs     = dq_bfs

    # Anchor: torso_link relative to reference torso in robot torso frame
    rob_tp = data.xpos[MJ_ANCHOR_ID]
    rob_tq = data.xquat[MJ_ANCHOR_ID]
    anchor_pos_b, anchor_qr = _subtract_frames(rob_tp, rob_tq,
                                               ref_torso_pos, ref_torso_quat)
    R_rel = _q2R(anchor_qr)
    anchor_ori_b = R_rel[:, :2].flatten().astype(np.float32)  # 6D, row-major

    return np.concatenate([
        ref_jp_bfs.astype(np.float32),
        ref_jv_bfs.astype(np.float32),
        anchor_pos_b.astype(np.float32),
        anchor_ori_b,
        base_lin_vel,
        base_ang_vel,
        joint_pos_rel_bfs,
        joint_vel_bfs,
        last_action_bfs.astype(np.float32),
    ])  # shape (160,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _best_start_frame(ref_jp_all_bfs: np.ndarray) -> int:
    """Return the NPZ frame whose joint positions are closest to DEFAULT_BFS.

    Starting from a frame near the training default pose means the policy
    sees joint_pos_rel values well within its training distribution (< 1σ),
    so leg motion is natural from step 0.  Frame 0 of most WBT clips has
    nearly straight knees (≈0.02 rad vs DEFAULT 0.67 rad), which pushes the
    required knee action to ~-8σ and causes the stiff-leg symptom.
    """
    # Weight legs (hip, knee, ankle) more than arms/wrists for stability
    weights = np.ones(29, np.float32)
    for j, name in enumerate(BFS_JOINT_NAMES):
        if any(k in name for k in ("hip", "knee", "ankle")):
            weights[j] = 4.0
    diff = ref_jp_all_bfs - DEFAULT_BFS[np.newaxis, :]
    err  = (diff ** 2 * weights[np.newaxis, :]).sum(axis=1)
    best = int(np.argmin(err))
    lk = ref_jp_all_bfs[best, BFS_JOINT_NAMES.index("left_knee_joint")]
    rk = ref_jp_all_bfs[best, BFS_JOINT_NAMES.index("right_knee_joint")]
    print(f"[sim2sim] Auto start frame: {best}  (t={best/50:.2f}s  "
          f"l_knee={lk:.3f} r_knee={rk:.3f}  err={err[best]:.4f})")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/wbt_g1_v1/model_29999.pt")
    parser.add_argument("--motion_file", default="/tmp/combined_g1_50fps.npz")
    parser.add_argument("--xml",
        default="/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml")
    parser.add_argument("--standing", action="store_true",
        help="Standing balance test: init at default pose, freeze reference there")
    parser.add_argument("--playback", action="store_true",
        help="Advance motion reference at 50 Hz")
    parser.add_argument("--frame", type=int, default=None,
        help="Starting NPZ frame index. Auto-selected (closest to default pose) "
             "when omitted in playback mode.")
    parser.add_argument("--sim_dt",       type=float, default=0.002)
    parser.add_argument("--decimation",   type=int,   default=10)
    parser.add_argument("--push_force",   type=float, default=200.0,
        help="Keyboard/auto push magnitude in N (default 200)")
    parser.add_argument("--push_duration",type=float, default=0.08,
        help="Push duration in seconds (default 0.08)")
    parser.add_argument("--push_interval",type=float, default=0.0,
        help="Auto-push interval in seconds; 0 = disabled (default 0)")
    args = parser.parse_args()

    actor = _build_actor(args.checkpoint)

    print(f"[sim2sim] Loading motion: {args.motion_file}")
    npz = np.load(args.motion_file)
    ref_jp_all_bfs  = npz["joint_pos"]    # (T, 29) BFS
    ref_jv_all_bfs  = npz["joint_vel"]    # (T, 29) BFS
    ref_bpos_all    = npz["body_pos_w"]   # (T, 30, 3)
    ref_bquat_all   = npz["body_quat_w"]  # (T, 30, 4) wxyz
    T_total = ref_jp_all_bfs.shape[0]
    fps = float(npz["fps"][0])
    print(f"[sim2sim] NPZ: {T_total} frames @ {fps:.0f} fps")

    action_scale_bfs, kp_dfs, kd_dfs, tau_max_dfs, arm_dfs = _make_pd_params()

    print(f"[sim2sim] Loading XML: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    model.opt.timestep = args.sim_dt

    # Patch model: set armature and implicit joint damping to match training.
    # Using MuJoCo's implicit damping (dof_damping) avoids explicit-kd instability.
    # Free joint occupies dof indices 0-5; revolute joints start at dof index 6.
    for i in range(29):
        model.dof_armature[i + 6] = arm_dfs[i]
        model.dof_damping[i + 6]  = kd_dfs[i]
    print("[sim2sim] Armature and implicit damping set from BeyondMimic G1 config")

    policy_dt = args.sim_dt * args.decimation
    if args.frame is not None:
        frame_idx = int(args.frame) % T_total
    elif args.playback:
        frame_idx = _best_start_frame(ref_jp_all_bfs)
    else:
        frame_idx = 0

    if args.standing:
        # Standing balance test: initialise from NPZ frame 0 (upright pose, CoM centred),
        # then freeze the reference at that same frame.
        # DEFAULT_DFS (bent knees) is not a stable equilibrium in g1_29dof.xml;
        # NPZ frame 0 has nearly straight legs and CoM ≈ 0 over the contact patch.
        data.qpos[0:3]  = ref_bpos_all[frame_idx, 0]
        data.qpos[3:7]  = ref_bquat_all[frame_idx, 0]
        data.qpos[7:36] = ref_jp_all_bfs[frame_idx][DFS_TO_BFS]  # BFS→DFS
        data.qvel[:]    = 0.0
        height_offset   = _snap_to_ground(model, data)
        # Standing mode: loop through a ±50-frame window around the best start frame
        # so the reference is gently varying (in-distribution) rather than frozen.
        # A perfectly frozen reference is out-of-distribution for the WBT policy
        # (trained with an always-advancing reference) and causes oscillation.
        _STAND_HALF = 50   # ±50 frames = ±1 second at 50 fps
        _stand_lo   = max(0, frame_idx - _STAND_HALF)
        _stand_hi   = min(T_total - 1, frame_idx + _STAND_HALF)
        _stand_frames = np.arange(_stand_lo, _stand_hi + 1)
        print(f"[sim2sim] Standing mode  centre={frame_idx}  "
              f"window=[{_stand_lo},{_stand_hi}] ({len(_stand_frames)} frames = "
              f"{len(_stand_frames)/fps:.1f} s loop)  pelvis_z={data.qpos[2]:.4f} m")
        ref_fk = RefFK(args.xml, ref_jp_all_bfs, ref_bpos_all, ref_bquat_all,
                       height_offset=height_offset)
        _stand_tick = [0]
        def get_ref(_frame):
            t = int(_stand_frames[_stand_tick[0] % len(_stand_frames)])
            _stand_tick[0] += 1
            tp, tq = ref_fk.get(t)
            return tp, tq, ref_jp_all_bfs[t].astype(np.float32), ref_jv_all_bfs[t].astype(np.float32)
    else:
        # NPZ playback or frozen NPZ frame
        data.qpos[0:3]  = ref_bpos_all[frame_idx, 0]
        data.qpos[3:7]  = ref_bquat_all[frame_idx, 0]
        data.qpos[7:36] = ref_jp_all_bfs[frame_idx][DFS_TO_BFS]  # BFS→DFS
        data.qvel[:]    = 0.0
        height_offset   = _snap_to_ground(model, data)
        ref_fk = RefFK(args.xml, ref_jp_all_bfs, ref_bpos_all, ref_bquat_all,
                       height_offset=height_offset)
        def get_ref(frame):
            tp, tq = ref_fk.get(frame)
            return tp, tq, ref_jp_all_bfs[frame % T_total], ref_jv_all_bfs[frame % T_total]

    print(f"[sim2sim] Height offset: {height_offset:+.4f} m  →  pelvis_z={data.qpos[2]:.4f}")

    mode = "standing" if args.standing else ("playback" if args.playback else "frozen NPZ")
    print(f"[sim2sim] Mode: {mode}   Policy @ {1/policy_dt:.0f} Hz")
    print("[sim2sim] Push keys: ↑↓←→ = directional push,  P = random push,  "
          "Ctrl+drag = mouse spring")

    last_action_bfs = np.zeros(29, np.float32)

    # -----------------------------------------------------------------------
    # Robustness push state (shared via closure with key_callback)
    # -----------------------------------------------------------------------
    _push_dur_steps = max(1, round(args.push_duration / args.sim_dt))
    _push = {'steps_left': 0, 'force': np.zeros(6, np.float64)}
    _auto_steps = [round(args.push_interval / policy_dt)] if args.push_interval > 0 else [0]
    _auto_counter = [0]

    # GLFW key codes (no glfw import needed — just integer constants)
    _KEY_UP    = 265
    _KEY_DOWN  = 264
    _KEY_LEFT  = 263
    _KEY_RIGHT = 262
    _KEY_P     = ord('P')  # 80

    def _trigger_push(dir_xyz: np.ndarray) -> None:
        """Apply a push of args.push_force N in dir_xyz (world frame, auto-normalised)."""
        n = np.linalg.norm(dir_xyz)
        if n < 1e-9:
            return
        _push['steps_left'] = _push_dur_steps
        _push['force'][:3]  = args.push_force * dir_xyz / n
        _push['force'][3:]  = 0.0
        print(f"[push] {args.push_force:.0f} N  dir={np.round(dir_xyz/n, 2)}  "
              f"dur={_push_dur_steps * args.sim_dt * 1000:.0f} ms")

    def key_callback(keycode: int) -> None:
        # Robot nominally faces +X in world frame
        if   keycode == _KEY_UP:    _trigger_push(np.array([ 1.,  0., 0.]))
        elif keycode == _KEY_DOWN:  _trigger_push(np.array([-1.,  0., 0.]))
        elif keycode == _KEY_LEFT:  _trigger_push(np.array([ 0.,  1., 0.]))
        elif keycode == _KEY_RIGHT: _trigger_push(np.array([ 0., -1., 0.]))
        elif keycode == _KEY_P:
            a = np.random.uniform(0.0, 2.0 * np.pi)
            _trigger_push(np.array([np.cos(a), np.sin(a), 0.0]))

    # -----------------------------------------------------------------------
    # Main simulation loop
    # -----------------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=key_callback) as viewer:
        while viewer.is_running():
            t_wall = time.perf_counter()

            # Auto-push: fire when counter reaches zero, then reset
            if args.push_interval > 0 and _push['steps_left'] == 0:
                _auto_counter[0] += 1
                if _auto_counter[0] >= _auto_steps[0]:
                    _auto_counter[0] = 0
                    a = np.random.uniform(0.0, 2.0 * np.pi)
                    _trigger_push(np.array([np.cos(a), np.sin(a), 0.0]))

            ref_tp, ref_tq, ref_jp_bfs, ref_jv_bfs = get_ref(frame_idx)

            obs = _build_obs(data, ref_jp_bfs, ref_jv_bfs,
                             ref_tp, ref_tq, last_action_bfs)
            with torch.inference_mode():
                action_bfs = actor(
                    torch.from_numpy(obs).unsqueeze(0)
                ).squeeze(0).numpy()
            last_action_bfs = action_bfs.copy()

            # Convert BFS action to DFS for MuJoCo; only stiffness term (kd is implicit)
            action_dfs = action_bfs[DFS_TO_BFS]
            target_q   = DEFAULT_DFS + action_scale_bfs[DFS_TO_BFS] * action_dfs

            for _ in range(args.decimation):
                # Apply external push force (world frame, on pelvis)
                if _push['steps_left'] > 0:
                    data.xfrc_applied[MJ_PELVIS_ID] = _push['force']
                    _push['steps_left'] -= 1
                else:
                    data.xfrc_applied[MJ_PELVIS_ID] = 0.0

                q  = data.qpos[7:36]
                dq = data.qvel[6:35]
                tau = kp_dfs * (target_q - q)   # kd handled by dof_damping
                np.clip(tau, -tau_max_dfs, tau_max_dfs, out=tau)
                data.ctrl[:] = tau
                mujoco.mj_step(model, data)

            viewer.sync()

            if args.playback:
                frame_idx += 1

            # Auto-reset when the robot falls (pelvis z < 0.3 m)
            if data.qpos[2] < 0.3:
                reset_frame = frame_idx % T_total
                print(f"[sim2sim] *** FELL — resetting to frame {reset_frame} ***")
                data.qpos[0:3]  = ref_bpos_all[reset_frame, 0]
                data.qpos[3:7]  = ref_bquat_all[reset_frame, 0]
                data.qpos[7:36] = ref_jp_all_bfs[reset_frame][DFS_TO_BFS]
                data.qvel[:]    = 0.0
                _snap_to_ground(model, data)
                last_action_bfs[:] = 0.0
                _push['steps_left'] = 0
                _push['force'][:]   = 0.0
                data.xfrc_applied[MJ_PELVIS_ID] = 0.0

            sleep_t = policy_dt - (time.perf_counter() - t_wall)
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
