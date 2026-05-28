"""
Sim2Sim: Load trained legged_lab AMP policy → run in MuJoCo.

Keyboard controls (when viewer is focused):
  W / ↑   forward faster    (+0.3 m/s)
  S / ↓   slow down         (-0.3 m/s)
  A / ←   turn left         (+0.4 rad/s)
  D / →   turn right        (-0.4 rad/s)
  Q       strafe left       (+0.2 m/s)
  E       strafe right      (-0.2 m/s)
  Space   stop (cmd → 0)
  R       reset robot pose
  Esc     quit

Usage:
  python sim2sim_g1_amp.py --checkpoint <path/to/model_N.pt>

Obs space (policy input, 585 = 117 per step × 5 history steps):
  base_ang_vel             3
  root_local_rot_tan_norm  6
  velocity_commands        3
  joint_pos               29  (USD / Isaac Lab joint order)
  joint_vel               29
  actions                 29
  key_body_pos_b          18  (6 bodies × 3, in pelvis frame)

Joint order matches the G1 29DOF USD file (Isaac Lab ordering):
  left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
  left_ankle_pitch, left_ankle_roll,
  right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
  right_ankle_pitch, right_ankle_roll,
  waist_yaw, waist_roll, waist_pitch,
  left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
  left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
  right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
  right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
"""

import argparse
import pickle
import threading
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────── paths ───────────────────────────────────────
G1_XML = "/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml"
# Motion data for reference-state initialization (wxyz quaternion, ISL_JOINTS dof order)
# Confirmed wxyz from legged_lab/managers/motion_data_manager.py line 99.
# Use mid-run frames (35-90) for stable initialization at walking speed.
MOTION_DATA_DIR = "/home/rexcon/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
MOTION_INIT_FILE = f"{MOTION_DATA_DIR}/C1_-_stand_to_run_stageii.pkl"  # 122 frames, 0→run
MOTION_INIT_FRAME_RANGE = (35, 90)   # mid-run frames; robot has forward velocity ≈ 0.5–2.5 m/s

# ──────────────────────────── joint ordering ──────────────────────────────────
# Must match the USD file order (Isaac Lab order), NOT alphabetical.
# Extracted from g1_29dof_rev_1_0.usd via USD traversal.
ISL_JOINTS = [
    "left_hip_pitch_joint",       # [0]
    "left_hip_roll_joint",        # [1]
    "left_hip_yaw_joint",         # [2]
    "left_knee_joint",            # [3]
    "left_ankle_pitch_joint",     # [4]
    "left_ankle_roll_joint",      # [5]
    "right_hip_pitch_joint",      # [6]
    "right_hip_roll_joint",       # [7]
    "right_hip_yaw_joint",        # [8]
    "right_knee_joint",           # [9]
    "right_ankle_pitch_joint",    # [10]
    "right_ankle_roll_joint",     # [11]
    "waist_yaw_joint",            # [12]
    "waist_roll_joint",           # [13]
    "waist_pitch_joint",          # [14]
    "left_shoulder_pitch_joint",  # [15]
    "left_shoulder_roll_joint",   # [16]
    "left_shoulder_yaw_joint",    # [17]
    "left_elbow_joint",           # [18]
    "left_wrist_roll_joint",      # [19]
    "left_wrist_pitch_joint",     # [20]
    "left_wrist_yaw_joint",       # [21]
    "right_shoulder_pitch_joint", # [22]
    "right_shoulder_roll_joint",  # [23]
    "right_shoulder_yaw_joint",   # [24]
    "right_elbow_joint",          # [25]
    "right_wrist_roll_joint",     # [26]
    "right_wrist_pitch_joint",    # [27]
    "right_wrist_yaw_joint",      # [28]
]
N_JOINTS = len(ISL_JOINTS)

# Default joint positions (USD order) matching Isaac Lab's InitialStateCfg
DEFAULT_POS_ISL = np.array([
    -0.1,   # left_hip_pitch
     0.0,   # left_hip_roll
     0.0,   # left_hip_yaw
     0.3,   # left_knee
    -0.2,   # left_ankle_pitch
     0.0,   # left_ankle_roll
    -0.1,   # right_hip_pitch
     0.0,   # right_hip_roll
     0.0,   # right_hip_yaw
     0.3,   # right_knee
    -0.2,   # right_ankle_pitch
     0.0,   # right_ankle_roll
     0.0,   # waist_yaw
     0.0,   # waist_roll
     0.0,   # waist_pitch
     0.3,   # left_shoulder_pitch
     0.25,  # left_shoulder_roll
     0.0,   # left_shoulder_yaw
     0.97,  # left_elbow
     0.15,  # left_wrist_roll
     0.0,   # left_wrist_pitch
     0.0,   # left_wrist_yaw
     0.3,   # right_shoulder_pitch
    -0.25,  # right_shoulder_roll
     0.0,   # right_shoulder_yaw
     0.97,  # right_elbow
    -0.15,  # right_wrist_roll
     0.0,   # right_wrist_pitch
     0.0,   # right_wrist_yaw
], dtype=np.float32)

# PD gains (USD order) from legged_lab unitree.py ImplicitActuatorCfg
KP_ISL = np.array([
    100, 100, 100, 150,  40,  40,   # left leg
    100, 100, 100, 150,  40,  40,   # right leg
    200,  40,  40,                   # waist (yaw, roll, pitch)
     40,  40,  40,  40,  40,  40,  40,  # left arm
     40,  40,  40,  40,  40,  40,  40,  # right arm
], dtype=np.float32)

KD_ISL = np.array([
      2,   2,   2,   4,   2,   2,   # left leg  (matches Isaac Lab ImplicitActuatorCfg)
      2,   2,   2,   4,   2,   2,   # right leg
      5,   5,   5,                   # waist
      1,   1,   1,   1,   1,   1,   1,  # left arm
      1,   1,   1,   1,   1,   1,   1,  # right arm
], dtype=np.float32)

EFFORT_LIMIT_ISL = np.array([
     88, 139,  88, 139,  25,  25,   # left leg
     88, 139,  88, 139,  25,  25,   # right leg
     88,  25,  25,                   # waist
     25,  25,  25,  25,  25,   5,   5,  # left arm
     25,  25,  25,  25,  25,   5,   5,  # right arm
], dtype=np.float32)

KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

ACTION_SCALE  = 0.25
OBS_PER_STEP  = 117   # 3+6+3+29+29+29+18
HISTORY_LEN   = 5
OBS_DIM       = OBS_PER_STEP * HISTORY_LEN  # 585

# Velocity command limits
VX_MIN, VX_MAX = -0.5, 3.0
VY_MIN, VY_MAX = -0.5, 0.5
WZ_MIN, WZ_MAX = -1.0, 1.0


# ───────────────────────────── math helpers ──────────────────────────────────

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])

def get_yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cy, 0.0, 0.0, sy])


# ─────────────────────────── minimal actor MLP ───────────────────────────────

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256),    nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_actor(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
    actor = Actor(OBS_DIM, N_JOINTS)
    actor.net.load_state_dict(actor_state)
    actor.eval()
    print(f"[sim2sim] Loaded iter={ckpt['iter']}  from {checkpoint_path}")
    return actor


# ─────────────────────────── MuJoCo setup ───────────────────────────────────

def build_model():
    model = mujoco.MjModel.from_xml_path(G1_XML)

    # Match Isaac Lab's physics timestep (sim.dt=0.005, decimation=4 → 50 Hz policy)
    model.opt.timestep = 0.005

    # Add armature to all joints to match Isaac Lab's ImplicitActuatorCfg(armature=0.01)
    model.dof_armature[:] = 0.01

    # More solver iterations to better approximate PhysX's implicit integration stability
    model.opt.iterations        = 50
    model.opt.noslip_iterations = 20

    isl_qpos_ids = np.array([model.joint(n).qposadr[0] for n in ISL_JOINTS], dtype=int)
    isl_dof_ids  = np.array([model.joint(n).dofadr[0]  for n in ISL_JOINTS], dtype=int)
    isl_act_ids  = np.array([model.actuator(n).id       for n in ISL_JOINTS], dtype=int)

    # Convert motor actuators to semi-implicit PD: ctrl = target_pos
    # force = kp*(ctrl - q) - kd*qd  (bias applied inside MuJoCo integrator)
    for idx, name in enumerate(ISL_JOINTS):
        aid = isl_act_ids[idx]
        kp  = float(KP_ISL[idx])
        kd  = float(KD_ISL[idx])
        ef  = float(EFFORT_LIMIT_ISL[idx])
        model.actuator_gaintype[aid]    = 0      # fixed gain
        model.actuator_gainprm[aid, 0]  = kp     # gain = kp → force = kp * ctrl
        model.actuator_biastype[aid]    = 1      # affine bias
        model.actuator_biasprm[aid, 0]  = 0.0   # constant term
        model.actuator_biasprm[aid, 1]  = -kp   # position feedback: -kp * q
        model.actuator_biasprm[aid, 2]  = -kd   # velocity feedback: -kd * qd
        model.actuator_forcerange[aid]  = [-ef, ef]

    key_body_ids = np.array([model.body(n).id for n in KEY_BODY_NAMES], dtype=int)
    pelvis_id    = model.body("pelvis").id
    # Joint limits in ISL_JOINTS order (for clipping position targets)
    jnt_lo = np.array([model.joint(n).range[0] for n in ISL_JOINTS], dtype=np.float32)
    jnt_hi = np.array([model.joint(n).range[1] for n in ISL_JOINTS], dtype=np.float32)
    return model, isl_qpos_ids, isl_dof_ids, isl_act_ids, key_body_ids, pelvis_id, jnt_lo, jnt_hi


def _load_motion():
    with open(MOTION_INIT_FILE, "rb") as f:
        d = pickle.load(f)
    return d

_MOTION_CACHE = None

def _wxyz_mul(q1, q2):
    """Quaternion multiply, both in wxyz format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _ang_vel_world(q0_wxyz, q1_wxyz, dt):
    """Approximate angular velocity in world frame from two wxyz quaternions."""
    dq = _wxyz_mul(q1_wxyz, np.array([q0_wxyz[0], -q0_wxyz[1], -q0_wxyz[2], -q0_wxyz[3]]))
    # ω_world = 2 * Im(dq * q0_conj) / dt  →  2 * dq[1:4] / dt  when |dq-identity| small
    return 2.0 * dq[1:4] / dt

def do_reset(model, data, isl_qpos_ids, isl_dof_ids, frame_idx=None):
    """Reset robot state from a reference motion frame.

    Motion data quaternion is in (w, x, y, z) wxyz format (confirmed from
    legged_lab/managers/motion_data_manager.py line 99).
    MuJoCo free-joint also uses wxyz — direct assignment after yaw removal.
    """
    global _MOTION_CACHE
    if _MOTION_CACHE is None:
        _MOTION_CACHE = _load_motion()
    m = _MOTION_CACHE
    fps      = float(m["fps"])
    root_pos = np.array(m["root_pos"])
    root_rot = np.array(m["root_rot"])   # wxyz  ← confirmed
    dof_pos  = np.array(m["dof_pos"])
    N        = root_pos.shape[0]

    if frame_idx is None:
        lo, hi = MOTION_INIT_FRAME_RANGE
        frame_idx = np.random.randint(lo, min(hi, N - 1))
    frame_idx = int(np.clip(frame_idx, 0, N - 2))
    next_idx  = frame_idx + 1

    # Motion data quaternion is already wxyz — just remove yaw so robot faces +x
    qw, qx, qy, qz = root_rot[frame_idx]   # wxyz
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cy, sy = np.cos(-yaw / 2.0), np.sin(-yaw / 2.0)
    # q_yaw_inv = (cy, 0, 0, -sy) in wxyz
    nw, nx, ny, nz = _wxyz_mul(
        np.array([cy, 0.0, 0.0, -sy]),
        np.array([qw, qx, qy, qz]),
    )

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, root_pos[frame_idx, 2]]
    data.qpos[3:7] = [nw, nx, ny, nz]   # wxyz — nearly identity for standing frames
    data.qpos[isl_qpos_ids] = dof_pos[frame_idx].astype(np.float32)

    # Linear velocity: rotate by -yaw so it aligns with +x facing direction
    dt_m = 1.0 / fps
    vel_world = (root_pos[next_idx] - root_pos[frame_idx]) / dt_m
    c, s = np.cos(-yaw), np.sin(-yaw)
    data.qvel[0] = c * vel_world[0] - s * vel_world[1]
    data.qvel[1] = s * vel_world[0] + c * vel_world[1]
    data.qvel[2] = vel_world[2]

    # Angular velocity from quaternion finite differences, rotated by -yaw
    aw = _ang_vel_world(root_rot[frame_idx], root_rot[next_idx], dt_m)
    data.qvel[3] = c * aw[0] - s * aw[1]
    data.qvel[4] = s * aw[0] + c * aw[1]
    data.qvel[5] = aw[2]

    # Joint velocities from finite differences
    data.qvel[isl_dof_ids] = ((dof_pos[next_idx] - dof_pos[frame_idx]) / dt_m).astype(np.float32)

    mujoco.mj_forward(model, data)


# ──────────────────────── observation assembly ───────────────────────────────

def get_obs_step(data, isl_qpos_ids, isl_dof_ids, key_body_ids, pelvis_id,
                 last_action, cmd):
    # Angular velocity: world→body frame
    ang_vel_w = data.qvel[3:6]
    R_pelvis  = data.xmat[pelvis_id].reshape(3, 3)
    ang_vel_b = R_pelvis.T @ ang_vel_w

    # Root local rotation (yaw-removed)
    root_quat    = data.qpos[3:7]  # wxyz
    yaw_quat     = get_yaw_quat(root_quat)
    local_quat   = quat_mul(quat_conjugate(yaw_quat), root_quat)
    R_local      = quat_to_rotmat(local_quat)
    rot_tan_norm = np.concatenate([R_local[:, 0], R_local[:, 2]])  # (6,)

    joint_pos = data.qpos[isl_qpos_ids].astype(np.float32)
    joint_vel = data.qvel[isl_dof_ids].astype(np.float32)

    root_pos_w = data.xpos[pelvis_id]
    key_pos_b  = np.concatenate([
        R_pelvis.T @ (data.xpos[bid] - root_pos_w) for bid in key_body_ids
    ]).astype(np.float32)

    return np.concatenate([
        ang_vel_b.astype(np.float32),   # 3
        rot_tan_norm.astype(np.float32), # 6
        cmd.astype(np.float32),          # 3
        joint_pos,                        # 29
        joint_vel,                        # 29
        last_action.astype(np.float32),  # 29
        key_pos_b,                        # 18
    ])  # total: 117


# ────────────────────────── keyboard state ───────────────────────────────────

# GLFW key codes
GLFW = {
    'W': 87, 'S': 83, 'A': 65, 'D': 68,
    'Q': 81, 'E': 69, 'R': 82,
    'UP': 265, 'DOWN': 264, 'LEFT': 263, 'RIGHT': 262,
    'SPACE': 32, 'ESC': 256,
}

class KeyboardCmd:
    def __init__(self, vx=0.5, vy=0.0, wz=0.0):
        self._cmd  = np.array([vx, vy, wz], dtype=np.float32)
        self._lock = threading.Lock()
        self._reset_flag = False
        self._quit_flag  = False

    def get(self):
        with self._lock:
            return self._cmd.copy()

    def pop_reset(self):
        with self._lock:
            v = self._reset_flag
            self._reset_flag = False
            return v

    def should_quit(self):
        with self._lock:
            return self._quit_flag

    def key_callback(self, keycode):
        with self._lock:
            vx, vy, wz = self._cmd
            if keycode in (GLFW['W'], GLFW['UP']):
                vx = min(vx + 0.3, VX_MAX)
            elif keycode in (GLFW['S'], GLFW['DOWN']):
                vx = max(vx - 0.3, VX_MIN)
            elif keycode in (GLFW['A'], GLFW['LEFT']):
                wz = min(wz + 0.4, WZ_MAX)
            elif keycode in (GLFW['D'], GLFW['RIGHT']):
                wz = max(wz - 0.4, WZ_MIN)
            elif keycode == GLFW['Q']:
                vy = min(vy + 0.2, VY_MAX)
            elif keycode == GLFW['E']:
                vy = max(vy - 0.2, VY_MIN)
            elif keycode == GLFW['SPACE']:
                vx, vy, wz = 0.0, 0.0, 0.0
            elif keycode == GLFW['R']:
                self._reset_flag = True
            elif keycode == GLFW['ESC']:
                self._quit_flag = True
                return
            self._cmd[:] = [vx, vy, wz]
            print(f"\r[cmd] vx={vx:+.1f}  vy={vy:+.1f}  wz={wz:+.1f}    ", end="", flush=True)


# ─────────────────────────────── main loop ───────────────────────────────────

def run(checkpoint, init_vx, init_vy, init_wz, max_steps):
    actor = load_actor(checkpoint)

    model, isl_qpos_ids, isl_dof_ids, isl_act_ids, key_body_ids, pelvis_id, jnt_lo, jnt_hi = build_model()
    data  = mujoco.MjData(model)

    kb = KeyboardCmd(vx=init_vx, vy=init_vy, wz=init_wz)

    obs_hist    = deque(maxlen=HISTORY_LEN)
    last_action = np.zeros(N_JOINTS, dtype=np.float32)
    target_pos  = DEFAULT_POS_ISL.copy()

    def reset_sim(frame_idx=None):
        nonlocal last_action, target_pos
        do_reset(model, data, isl_qpos_ids, isl_dof_ids, frame_idx=frame_idx)
        last_action = np.zeros(N_JOINTS, dtype=np.float32)
        target_pos  = data.qpos[isl_qpos_ids].copy()  # hold current ref pose
        obs_hist.clear()
        cmd = kb.get()
        init_obs = get_obs_step(data, isl_qpos_ids, isl_dof_ids, key_body_ids,
                                pelvis_id, last_action, cmd)
        for _ in range(HISTORY_LEN):
            obs_hist.append(init_obs.copy())

    reset_sim()

    POLICY_DT      = 0.02   # 50 Hz, matching Isaac Lab decimation=4 at dt=0.005
    SIM_PER_POLICY = max(1, int(round(POLICY_DT / model.opt.timestep)))

    print(f"[sim2sim] SIM_DT={model.opt.timestep:.4f}  POLICY_DT={POLICY_DT:.3f}  steps/policy={SIM_PER_POLICY}")
    SIM_DT = model.opt.timestep  # ensure local var reflects model
    print("[sim2sim] Controls: W/S=speed  A/D=turn  Q/E=strafe  Space=stop  R=reset  Esc=quit")
    print(f"[sim2sim] Init cmd: vx={init_vx}  vy={init_vy}  wz={init_wz}")

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=kb.key_callback)
    viewer.cam.distance  = 3.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth   = 90

    step_count  = 0
    policy_step = 0
    t_start     = time.time()

    try:
        while step_count < max_steps and viewer.is_running() and not kb.should_quit():

            if kb.pop_reset():
                print("\n[sim2sim] Manual reset")
                reset_sim()

            # ── PD: send target position; MuJoCo computes force internally ──
            data.ctrl[isl_act_ids] = target_pos
            mujoco.mj_step(model, data)
            step_count += 1

            # ── At policy rate: observe (post-step) then infer ────────────────
            if step_count % SIM_PER_POLICY == 0:
                cmd = kb.get()
                # Append obs computed from state AFTER the physics steps
                obs_hist.append(get_obs_step(
                    data, isl_qpos_ids, isl_dof_ids, key_body_ids,
                    pelvis_id, last_action, cmd,
                ))

                obs_vec = np.concatenate(list(obs_hist))  # oldest→newest
                obs_t   = torch.from_numpy(obs_vec).unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_t).squeeze(0).numpy()
                # last_action in obs must be the raw policy output (matches training)
                last_action = action.copy()
                # Clip to joint limits so actuators don't fight constraint forces
                target_pos  = np.clip(DEFAULT_POS_ISL + ACTION_SCALE * action, jnt_lo, jnt_hi)
                policy_step += 1

            # Camera follows robot
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            # Fall / NaN detection → auto-reset
            if data.qpos[2] < 0.3 or not np.isfinite(data.qpos).all():
                print(f"\n[sim2sim] Fell at policy_step={policy_step}  t={data.time:.2f}s — resetting")
                reset_sim()

            if policy_step % 200 == 0 and step_count % SIM_PER_POLICY == 0:
                cmd = kb.get()
                print(f"\r  t={data.time:6.1f}s  h={data.qpos[2]:.2f}m  "
                      f"vel_x={data.qvel[0]:+.2f}  "
                      f"cmd=[{cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f}]    ",
                      end="", flush=True)

    finally:
        viewer.close()

    elapsed = time.time() - t_start
    print(f"\n[sim2sim] Done. {policy_step} policy steps in {elapsed:.1f}s")


# ─────────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    DEFAULT_CKPT = (
        "/home/rexcon/legged_lab/scripts/rsl_rl/logs/rsl_rl/g1_amp/"
        "2026-05-25_20-59-53/model_100398.pt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--cmd_vx",  type=float, default=0.5)
    parser.add_argument("--cmd_vy",  type=float, default=0.0)
    parser.add_argument("--cmd_wz",  type=float, default=0.0)
    parser.add_argument("--steps",   type=int,   default=500000)
    args = parser.parse_args()

    run(
        checkpoint=args.checkpoint,
        init_vx=args.cmd_vx,
        init_vy=args.cmd_vy,
        init_wz=args.cmd_wz,
        max_steps=args.steps,
    )
