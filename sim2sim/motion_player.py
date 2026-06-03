"""
Motion Player: execute a retargeted joint-angle trajectory in MuJoCo G1.

Input:  NPZ file with keys:
          dof_positions   (T, 29)  float32, ISL joint order
          dof_velocities  (T, 29)  float32, optional
          root_positions  (T, 3)   float32, optional
          root_rotations  (T, 4)   float32 wxyz, optional
          fps             scalar

Output: NPZ file with keys:
          target_dof      (T, 29)  the input targets
          actual_dof      (T, 29)  measured dof_pos after each step
          geo_error       (T,)     mean |actual - target| per frame
          screenshots     list of PNG bytes at keyframes (every SCREENSHOT_EVERY frames)

Usage:
  python motion_player.py --traj <path.npz> [--out <result.npz>] [--headless]

Keyboard (viewer mode):
  R    restart from frame 0
  Esc  quit
"""

import argparse
import io
import time

import mujoco
import mujoco.viewer
import numpy as np

# ── Shared constants from sim2sim_g1_amp.py ───────────────────────────────────

G1_XML = "/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml"

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

DEFAULT_POS_ISL = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
     0.0, 0.0, 0.0,                     # waist
     0.3,  0.25, 0.0, 0.97,  0.15, 0.0, 0.0,  # left arm
     0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,  # right arm
], dtype=np.float32)

KP_ISL = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    200,  40,  40,
     40,  40,  40, 40, 40, 40, 40,
     40,  40,  40, 40, 40, 40, 40,
], dtype=np.float32)

KD_ISL = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    5, 5, 5,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
], dtype=np.float32)

EFFORT_LIMIT_ISL = np.array([
     88, 139,  88, 139, 25, 25,
     88, 139,  88, 139, 25, 25,
     88,  25,  25,
     25,  25,  25, 25, 25,  5,  5,
     25,  25,  25, 25, 25,  5,  5,
], dtype=np.float32)

SCREENSHOT_EVERY = 30   # capture every N trajectory frames


# ── MuJoCo model setup ────────────────────────────────────────────────────────

def build_model():
    model = mujoco.MjModel.from_xml_path(G1_XML)
    model.opt.timestep   = 0.005
    model.dof_armature[:]= 0.01
    model.opt.iterations        = 50
    model.opt.noslip_iterations = 20

    isl_qpos_ids = np.array([model.joint(n).qposadr[0] for n in ISL_JOINTS], dtype=int)
    isl_dof_ids  = np.array([model.joint(n).dofadr[0]  for n in ISL_JOINTS], dtype=int)
    isl_act_ids  = np.array([model.actuator(n).id       for n in ISL_JOINTS], dtype=int)

    jnt_lo = np.array([model.joint(n).range[0] for n in ISL_JOINTS], dtype=np.float32)
    jnt_hi = np.array([model.joint(n).range[1] for n in ISL_JOINTS], dtype=np.float32)

    for idx in range(N_JOINTS):
        aid = isl_act_ids[idx]
        kp, kd, ef = float(KP_ISL[idx]), float(KD_ISL[idx]), float(EFFORT_LIMIT_ISL[idx])
        model.actuator_gaintype[aid]   = 0
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biastype[aid]   = 1
        model.actuator_biasprm[aid, 0] = 0.0
        model.actuator_biasprm[aid, 1] = -kp
        model.actuator_biasprm[aid, 2] = -kd
        model.actuator_forcerange[aid] = [-ef, ef]

    return model, isl_qpos_ids, isl_dof_ids, isl_act_ids, jnt_lo, jnt_hi


def reset_to_frame(model, data, isl_qpos_ids, target_dof, frame_idx,
                   root_positions=None, root_rotations=None):
    mujoco.mj_resetData(model, data)
    if root_positions is not None:
        data.qpos[0:3] = root_positions[frame_idx]
    else:
        data.qpos[2] = 0.78   # standing height fallback
    if root_rotations is not None:
        data.qpos[3:7] = root_rotations[frame_idx]   # wxyz
    else:
        data.qpos[3] = 1.0   # identity quaternion
    data.qpos[isl_qpos_ids] = target_dof[frame_idx]
    mujoco.mj_forward(model, data)


# ── Trajectory execution ──────────────────────────────────────────────────────

def run(traj_path: str, out_path: str, headless: bool):
    data_npz   = np.load(traj_path, allow_pickle=True)
    target_dof = data_npz["dof_positions"].astype(np.float32)   # (T, 29) ISL order
    traj_fps   = float(data_npz.get("fps", 30.0))

    root_pos = data_npz["root_positions"] if "root_positions" in data_npz else None
    root_rot = data_npz["root_rotations"] if "root_rotations" in data_npz else None

    T = target_dof.shape[0]
    print(f"[player] Trajectory: {T} frames @ {traj_fps:.1f} fps  ({T/traj_fps:.2f}s)")

    model, isl_qpos_ids, isl_dof_ids, isl_act_ids, jnt_lo, jnt_hi = build_model()
    data = mujoco.MjData(model)

    # How many physics steps to hold each trajectory frame
    POLICY_DT      = 1.0 / traj_fps
    SIM_PER_FRAME  = max(1, int(round(POLICY_DT / model.opt.timestep)))
    print(f"[player] SIM_DT={model.opt.timestep}  traj_dt={POLICY_DT:.4f}  sim_steps/frame={SIM_PER_FRAME}")

    actual_dof  = np.zeros((T, N_JOINTS), dtype=np.float32)
    geo_error   = np.zeros(T, dtype=np.float32)
    screenshots = []

    renderer = mujoco.Renderer(model, height=480, width=640) if headless else None

    def capture_frame():
        if headless:
            renderer.update_scene(data, camera="tracking_camera" if "tracking_camera" in
                                  [model.camera(i).name for i in range(model.ncam)] else -1)
            img = renderer.render()
        else:
            img = None   # viewer screenshots handled separately
        return img

    reset_flag = [False]
    quit_flag  = [False]

    def key_callback(keycode):
        if keycode == 256:   # Esc
            quit_flag[0] = True
        elif keycode == 82:  # R
            reset_flag[0] = True

    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 90

    frame_idx = 0
    reset_to_frame(model, data, isl_qpos_ids, target_dof, 0, root_pos, root_rot)

    print("[player] Running...  (R=restart  Esc=quit)")

    while frame_idx < T:
        if not headless and quit_flag[0]:
            break
        if not headless and reset_flag[0]:
            reset_flag[0] = False
            frame_idx = 0
            actual_dof[:] = 0
            geo_error[:]  = 0
            screenshots.clear()
            reset_to_frame(model, data, isl_qpos_ids, target_dof, 0, root_pos, root_rot)
            print("\n[player] Restarted")
            continue

        # Clamp target to joint limits
        ctrl = np.clip(target_dof[frame_idx], jnt_lo, jnt_hi)
        data.ctrl[isl_act_ids] = ctrl

        for _ in range(SIM_PER_FRAME):
            mujoco.mj_step(model, data)

        # Record
        actual_dof[frame_idx] = data.qpos[isl_qpos_ids]
        geo_error[frame_idx]  = float(np.mean(np.abs(actual_dof[frame_idx] - target_dof[frame_idx])))

        # Screenshot every N frames
        if frame_idx % SCREENSHOT_EVERY == 0:
            img = capture_frame()
            if img is not None:
                from PIL import Image
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")
                screenshots.append(buf.getvalue())
            progress = frame_idx / T * 100
            print(f"\r  frame {frame_idx:4d}/{T}  geo_err={geo_error[frame_idx]:.4f}  {progress:.0f}%    ",
                  end="", flush=True)

        if not headless:
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

        # Fall detection
        if data.qpos[2] < 0.2 or not np.isfinite(data.qpos).all():
            print(f"\n[player] Fell at frame {frame_idx} — stopping early")
            actual_dof[frame_idx:] = actual_dof[frame_idx]
            geo_error[frame_idx:]  = geo_error[frame_idx]
            break

        frame_idx += 1

    if not headless:
        viewer.close()
    if renderer is not None:
        renderer.close()

    print(f"\n[player] Done. Mean geo_error={geo_error[:frame_idx].mean():.4f} rad")
    print(f"         Worst frame: {geo_error[:frame_idx].argmax()} "
          f"(err={geo_error[:frame_idx].max():.4f})")

    np.savez(out_path,
             target_dof=target_dof,
             actual_dof=actual_dof,
             geo_error=geo_error,
             screenshots=np.array(screenshots, dtype=object) if screenshots else np.array([]),
             fps=traj_fps)
    print(f"[player] Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",     required=True, help="Input NPZ trajectory file")
    parser.add_argument("--out",      default="motion_result.npz", help="Output NPZ file")
    parser.add_argument("--headless", action="store_true", help="No viewer, render offscreen")
    args = parser.parse_args()
    run(args.traj, args.out, args.headless)
