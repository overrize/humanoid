"""Generate a supine→standing get-up reference motion in BeyondMimic NPZ format.

Strategy (Path 1, BeyondMimic flavour):
  - No MoCap needed.  We define 7 kinematic keyframes (joint angles + pelvis quat),
    run MuJoCo FK to get all body poses, snap every frame to the ground, then
    interpolate at 50 fps using cubic splines (joints) and SLERP (quaternions).
  - The resulting NPZ is a valid BeyondMimic motion clip.  When the WBT policy is
    trained on it, the episode always starts with the robot lying flat (matching
    the first reference frame), and the tracking rewards guide it upward.
  - Termination (anchor_pos_z_only, 0.25 m) naturally paces learning: the policy
    first masters the early tucking phase, then is pushed further by adaptive
    sampling until the full get-up is learned.

Outputs:
  /tmp/getup_g1_50fps.npz   (ready to concatenate with combined_g1_50fps.npz)
"""

import math
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


# ── constants ──────────────────────────────────────────────────────────────────
XML = '/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml'
FPS = 50

# DFS joint order (matches MuJoCo qpos[7:36])
DFS = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",         "waist_roll_joint",        "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",        "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",       "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
DFS_IDX = {n: i for i, n in enumerate(DFS)}

# BFS joint order (NPZ joint_pos order)
BFS = [
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
BFS_IDX = {n: i for i, n in enumerate(BFS)}
BFS_TO_DFS = np.array([DFS_IDX[n] for n in BFS], dtype=int)
DFS_TO_BFS = np.zeros(29, dtype=int)
for j, i in enumerate(BFS_TO_DFS):
    DFS_TO_BFS[i] = j

# BFS body order: NPZ body index → MuJoCo body index
# (BFS traversal from body 1=pelvis; verified against existing NPZ)
BFS_MJ = np.array([
     1,  2,  8, 14,   # pelvis, L-hip-pitch, R-hip-pitch, waist-yaw
     3,  9, 15,        # L-hip-roll, R-hip-roll, waist-roll
     4, 10, 16,        # L-hip-yaw,  R-hip-yaw,  torso-link
     5, 11, 17, 24,   # L-knee, R-knee, L-sho-pitch, R-sho-pitch
     6, 12, 18, 25,   # L-ankle-pitch, R-ankle-pitch, L-sho-roll, R-sho-roll
     7, 13, 19, 26,   # L-ankle-roll,  R-ankle-roll,  L-sho-yaw,  R-sho-yaw
    20, 27,            # L-elbow, R-elbow
    21, 28,            # L-wrist-roll, R-wrist-roll
    22, 29,            # L-wrist-pitch, R-wrist-pitch
    23, 30,            # L-wrist-yaw, R-wrist-yaw
], dtype=int)

# Default joint positions (training default, DFS order)
_JP_DEFAULT = {
    "left_hip_pitch_joint":  -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint":        0.669, "right_knee_joint":       0.669,
    "left_ankle_pitch_joint":-0.363, "right_ankle_pitch_joint":-0.363,
    "left_elbow_joint":       0.600, "right_elbow_joint":      0.600,
    "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint":-0.2,"right_shoulder_pitch_joint": 0.2,
}
DEFAULT_DFS = np.array([_JP_DEFAULT.get(j, 0.0) for j in DFS], np.float32)


# ── helpers ────────────────────────────────────────────────────────────────────

def _snap_z(model: mujoco.MjModel, data: mujoco.MjData, margin: float = 0.002) -> float:
    """Raise qpos[2] so the lowest robot geom is at z=margin. Returns offset."""
    mujoco.mj_forward(model, data)
    low = np.inf
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0:
            continue
        gt  = model.geom_type[i]
        pos = data.geom_xpos[i]
        sz  = model.geom_size[i]
        mat = data.geom_xmat[i].reshape(3, 3)
        if gt == 2:    # sphere
            b = pos[2] - sz[0]
        elif gt == 7:  # capsule
            b = pos[2] - sz[0] - sz[1] * abs(mat[2, 2])
        else:
            b = pos[2] - max(sz[0], sz[1] if sz.size > 1 else sz[0])
        if b < low:
            low = b
    corr = margin - low
    data.qpos[2] += corr
    mujoco.mj_forward(model, data)
    return corr


def _extract_bodies(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Extract body poses in BFS (NPZ) order. Returns (pos(30,3), quat(30,4) wxyz)."""
    pos  = data.xpos[BFS_MJ].copy().astype(np.float32)
    quat = data.xquat[BFS_MJ].copy().astype(np.float32)   # MuJoCo xquat is wxyz
    return pos, quat


def _quat_wxyz_to_scipy(q: np.ndarray) -> Rotation:
    """Convert wxyz → scipy xyzw Rotation."""
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def _scipy_to_quat_wxyz(r: Rotation) -> np.ndarray:
    xyzw = r.as_quat()
    if xyzw.ndim == 1:
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], np.float32)
    return np.stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]], axis=1).astype(np.float32)


def _ang_vel_from_quats(q: np.ndarray, dt: float) -> np.ndarray:
    """Approximate angular velocity (world frame) from consecutive wxyz quats. (T, 4) → (T, 3)."""
    T = q.shape[0]
    w = np.zeros((T, 3), np.float32)
    for t in range(T - 1):
        r0 = _quat_wxyz_to_scipy(q[t])
        r1 = _quat_wxyz_to_scipy(q[t + 1])
        dR = r1 * r0.inv()
        rotvec = dR.as_rotvec()
        w[t] = (rotvec / dt).astype(np.float32)
    w[-1] = w[-2]
    return w


# ── keyframe definition ────────────────────────────────────────────────────────
#
# Each keyframe: t (s), pelvis_quat (wxyz), joint_angles (dict, DFS names)
# Pelvis x,y = 0 throughout; z is snapped to ground after FK.
#
# Supine orientation: [0.707, 0, -0.707, 0]
#   Robot chest faces world +Z (up). Legs extend in world +X.
#   Confirmed by FK: pelvis z = 0.142 m when snapped to ground.
#
# Rise sequence:
#   0.0 s – lying flat, all joints 0
#   1.0 s – knees tuck while still supine
#   2.2 s – pelvis 45° between supine and upright ("rolling")
#   3.2 s – pelvis upright, deep squat / kneeling stance
#   4.2 s – pelvis upright, shallow squat
#   5.2 s – standing (DEFAULT joints)
#   7.0 s – standing hold

def _joints(d: dict) -> np.ndarray:
    """Build a DFS joint angle array from a partial dict (others = 0)."""
    out = np.zeros(29, np.float32)
    for name, val in d.items():
        out[DFS_IDX[name]] = val
    return out


# shoulders slightly open to give balance during transition
_SHOULDER_BRACE = {
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint":  0.4,
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.4,
}

KEYFRAMES = [
    # t, pelvis_quat_wxyz, joint_angles_dfs
    (0.0,
     np.array([0.707, 0.0, -0.707, 0.0], np.float32),
     _joints({})),

    (1.0,
     np.array([0.707, 0.0, -0.707, 0.0], np.float32),
     _joints({"left_hip_pitch_joint": -1.2, "right_hip_pitch_joint": -1.2,
              "left_knee_joint": 1.5, "right_knee_joint": 1.5,
              **_SHOULDER_BRACE})),

    (2.2,
     np.array([0.924, 0.0, -0.383, 0.0], np.float32),   # ~45° toward upright
     _joints({"left_hip_pitch_joint": -1.0, "right_hip_pitch_joint": -1.0,
              "left_knee_joint": 1.3, "right_knee_joint": 1.3,
              **_SHOULDER_BRACE})),

    (3.2,
     np.array([1.0, 0.0, 0.0, 0.0], np.float32),
     _joints({"left_hip_pitch_joint": -1.1, "right_hip_pitch_joint": -1.1,
              "left_knee_joint": 1.4, "right_knee_joint": 1.4,
              "left_ankle_pitch_joint": -0.3, "right_ankle_pitch_joint": -0.3,
              **_SHOULDER_BRACE})),

    (4.2,
     np.array([1.0, 0.0, 0.0, 0.0], np.float32),
     _joints({"left_hip_pitch_joint": -0.55, "right_hip_pitch_joint": -0.55,
              "left_knee_joint": 0.95, "right_knee_joint": 0.95,
              "left_ankle_pitch_joint": -0.35, "right_ankle_pitch_joint": -0.35,
              "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2,
              "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2})),

    (5.2,
     np.array([1.0, 0.0, 0.0, 0.0], np.float32),
     DEFAULT_DFS.copy()),

    (7.0,
     np.array([1.0, 0.0, 0.0, 0.0], np.float32),
     DEFAULT_DFS.copy()),
]


# ── generate frames ────────────────────────────────────────────────────────────

def generate(output_path: str = "/tmp/getup_g1_50fps.npz") -> None:
    model = mujoco.MjModel.from_xml_path(XML)
    data  = mujoco.MjData(model)

    # 1. Evaluate every keyframe to get snapped pelvis z and body poses
    kf_times  = [kf[0] for kf in KEYFRAMES]
    kf_quats  = [kf[1] for kf in KEYFRAMES]   # wxyz
    kf_joints = [kf[2] for kf in KEYFRAMES]   # DFS

    kf_pelvis_z   = []
    kf_body_pos   = []
    kf_body_quat  = []

    print("Evaluating keyframes:")
    for idx, (t, pq, jd) in enumerate(KEYFRAMES):
        data.qpos[0:3]  = [0.0, 0.0, 0.8]   # starting z guess
        data.qpos[3:7]  = pq
        data.qpos[7:36] = jd
        data.qvel[:]    = 0.0
        corr = _snap_z(model, data)
        z = float(data.qpos[2])
        kf_pelvis_z.append(z)
        bp, bq = _extract_bodies(data)
        kf_body_pos.append(bp)
        kf_body_quat.append(bq)
        # Print torso z and lowest geom
        torso_mj = 16
        print(f"  KF{idx}  t={t:.1f}s  pelvis_z={z:.3f}  "
              f"torso_z={data.xpos[torso_mj,2]:.3f}  snap={corr:+.3f}")

    # 2. Build time axis at FPS
    T_end   = kf_times[-1]
    n_frame = int(round(T_end * FPS)) + 1
    t_arr   = np.linspace(0.0, T_end, n_frame)
    dt      = 1.0 / FPS

    # 3. Interpolate joint angles (cubic spline, DFS order)
    kf_joints_arr = np.stack(kf_joints, axis=0)   # (nkf, 29)
    cs_joints = CubicSpline(kf_times, kf_joints_arr)
    joints_dfs = cs_joints(t_arr).astype(np.float32)   # (T, 29)
    joints_bfs = joints_dfs[:, BFS_TO_DFS]             # (T, 29) BFS for NPZ

    # Joint velocities via spline derivative
    jvel_dfs = cs_joints(t_arr, 1).astype(np.float32)
    jvel_bfs = jvel_dfs[:, BFS_TO_DFS]

    # 4. Interpolate pelvis z (linear is fine)
    from scipy.interpolate import interp1d
    cs_z = interp1d(kf_times, kf_pelvis_z, kind='cubic')
    pelvis_z = cs_z(t_arr).astype(np.float32)          # (T,)

    # 5. SLERP pelvis quaternion
    kf_rot = _quat_wxyz_to_scipy(np.stack(kf_quats))
    slerp  = Slerp(kf_times, kf_rot)
    pelvis_quat_wxyz = _scipy_to_quat_wxyz(slerp(t_arr))  # (T, 4) wxyz

    # 6. Re-run FK for every frame to get exact body poses
    print(f"\nRunning FK for {n_frame} frames…")
    body_pos_all  = np.zeros((n_frame, 30, 3),  np.float32)
    body_quat_all = np.zeros((n_frame, 30, 4),  np.float32)

    for f in range(n_frame):
        data.qpos[0]    = 0.0
        data.qpos[1]    = 0.0
        data.qpos[2]    = float(pelvis_z[f])
        data.qpos[3:7]  = pelvis_quat_wxyz[f]
        data.qpos[7:36] = joints_dfs[f]
        data.qvel[:]    = 0.0
        mujoco.mj_forward(model, data)
        bp, bq = _extract_bodies(data)
        body_pos_all[f]  = bp
        body_quat_all[f] = bq

    # 7. Body linear velocities: finite differences of body_pos
    body_lin_vel = np.zeros_like(body_pos_all)
    body_lin_vel[:-1] = (body_pos_all[1:] - body_pos_all[:-1]) / dt
    body_lin_vel[-1]  = body_lin_vel[-2]

    # 8. Body angular velocities: from consecutive quaternion pairs
    body_ang_vel = np.zeros_like(body_pos_all)
    for b in range(30):
        body_ang_vel[:, b] = _ang_vel_from_quats(body_quat_all[:, b], dt)

    # 9. Save
    np.savez(
        output_path,
        fps           = np.array([FPS], dtype=np.int64),
        joint_pos     = joints_bfs,
        joint_vel     = jvel_bfs,
        body_pos_w    = body_pos_all,
        body_quat_w   = body_quat_all,
        body_lin_vel_w= body_lin_vel,
        body_ang_vel_w= body_ang_vel,
    )
    print(f"\nSaved → {output_path}")
    print(f"  frames={n_frame}  duration={T_end:.1f}s  fps={FPS}")
    print(f"  joint_pos shape: {joints_bfs.shape}")
    print(f"  body_pos_w shape: {body_pos_all.shape}")
    print(f"\nKey body z at start/end:")
    torso_npz = list(BFS_MJ).index(16)   # NPZ index for torso_link
    print(f"  frame  0  torso_z={body_pos_all[0,  torso_npz, 2]:.3f}")
    print(f"  frame -1  torso_z={body_pos_all[-1, torso_npz, 2]:.3f}")
    print(f"  pelvis_z: {body_pos_all[0,0,2]:.3f} → {body_pos_all[-1,0,2]:.3f}")


# ── combine with existing NPZ ──────────────────────────────────────────────────

def combine(getup_path: str, existing_path: str, output_path: str) -> None:
    """Concatenate get-up NPZ with the existing locomotion NPZ."""
    a = np.load(existing_path)
    b = np.load(getup_path)
    keys = list(a.keys())
    merged = {}
    for k in keys:
        if k == "fps":
            merged[k] = a[k]
        else:
            merged[k] = np.concatenate([a[k], b[k]], axis=0)
    np.savez(output_path, **merged)
    print(f"Combined: {a['joint_pos'].shape[0]} + {b['joint_pos'].shape[0]} "
          f"= {merged['joint_pos'].shape[0]} frames → {output_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="/tmp/getup_g1_50fps.npz")
    parser.add_argument("--combine",  action="store_true",
                        help="Also combine with existing locomotion NPZ")
    parser.add_argument("--existing", default="/tmp/combined_g1_50fps.npz")
    parser.add_argument("--combined_out", default="/tmp/combined_with_getup_g1_50fps.npz")
    args = parser.parse_args()

    generate(args.output)
    if args.combine:
        combine(args.output, args.existing, args.combined_out)
