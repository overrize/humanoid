"""Convert legged_lab deepmimic PKL files to BeyondMimic NPZ format.

The PKL format (legged_lab) stores: fps, root_pos(N,3), root_rot(N,4 wxyz),
dof_pos(N,29), key_body_pos(N,6,3), loop_mode.

The NPZ format (whole_body_tracking) stores: fps, joint_pos(N,29),
joint_vel(N,29), body_pos_w(N,B,3), body_quat_w(N,B,4),
body_lin_vel_w(N,B,3), body_ang_vel_w(N,B,3) for ALL B robot bodies.

FK is computed via Isaac Lab simulation (same approach as csv_to_npz.py).

Usage:
    # Single file
    python pkl_to_npz.py --input /path/to/motion.pkl --output /path/to/motion.npz

    # Directory (batch convert all PKL files)
    python pkl_to_npz.py --input_dir /path/to/pkls/ --output_dir /path/to/npzs/
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert legged_lab PKL to BeyondMimic NPZ via Isaac Lab FK.")
parser.add_argument("--input", type=str, default=None, help="Single PKL input file.")
parser.add_argument("--output", type=str, default=None, help="Single NPZ output file.")
parser.add_argument("--input_dir", type=str, default=None, help="Directory of PKL files (batch mode).")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to write NPZ files (batch mode).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import numpy as np
import joblib
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def convert_pkl_to_npz(pkl_path: str, npz_path: str, sim: SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    # Load PKL
    data = joblib.load(pkl_path)
    fps = int(data["fps"])
    dt = 1.0 / fps
    root_pos = torch.tensor(data["root_pos"], dtype=torch.float32, device=sim.device)   # (N, 3)
    root_rot = torch.tensor(data["root_rot"], dtype=torch.float32, device=sim.device)   # (N, 4) wxyz
    dof_pos = torch.tensor(data["dof_pos"], dtype=torch.float32, device=sim.device)     # (N, 29)
    N = root_pos.shape[0]

    log_joint_pos = []
    log_body_pos_w = []
    log_body_quat_w = []

    for i in range(N):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos[i]
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = root_rot[i]
        root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(root_state)

        jp = robot.data.default_joint_pos.clone()
        jv = robot.data.default_joint_vel.clone()
        jp[0] = dof_pos[i]
        jv[0] = 0.0
        robot.write_joint_state_to_sim(jp, jv)

        sim.render()
        scene.update(dt)

        log_joint_pos.append(robot.data.joint_pos[0].cpu().numpy().copy())
        log_body_pos_w.append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log_body_quat_w.append(robot.data.body_quat_w[0].cpu().numpy().copy())

    joint_pos = np.stack(log_joint_pos, axis=0)    # (N, 29)
    body_pos_w = np.stack(log_body_pos_w, axis=0)  # (N, B, 3)
    body_quat_w = np.stack(log_body_quat_w, axis=0)  # (N, B, 4) wxyz

    # Strip env origin offset from body positions so the motion is origin-relative
    origin_xy = scene.env_origins[0, :2].cpu().numpy()
    body_pos_w[:, :, :2] -= origin_xy

    # Compute velocities via finite differences
    joint_pos_t = torch.tensor(joint_pos, dtype=torch.float32)
    body_pos_t = torch.tensor(body_pos_w, dtype=torch.float32)
    body_quat_t = torch.tensor(body_quat_w, dtype=torch.float32)

    joint_vel = torch.gradient(joint_pos_t, spacing=dt, dim=0)[0].numpy()
    body_lin_vel_w = torch.gradient(body_pos_t, spacing=dt, dim=0)[0].numpy()
    body_ang_vel_w = _so3_derivative(body_quat_t, dt).numpy()

    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez(
        npz_path,
        fps=np.array([fps]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"[pkl_to_npz] Saved {N} frames → {npz_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 50  # render only, physics dt doesn't matter
    sim = SimulationContext(sim_cfg)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    if args_cli.input and args_cli.output:
        convert_pkl_to_npz(args_cli.input, args_cli.output, sim, scene)
    elif args_cli.input_dir and args_cli.output_dir:
        pkl_files = sorted(f for f in os.listdir(args_cli.input_dir) if f.endswith(".pkl"))
        print(f"[pkl_to_npz] Found {len(pkl_files)} PKL files in {args_cli.input_dir}")
        for pkl_file in pkl_files:
            in_path = os.path.join(args_cli.input_dir, pkl_file)
            out_path = os.path.join(args_cli.output_dir, pkl_file.replace(".pkl", ".npz"))
            if os.path.exists(out_path):
                print(f"[pkl_to_npz] Skipping {pkl_file} (already exists)")
                continue
            print(f"[pkl_to_npz] Converting {pkl_file} ...")
            convert_pkl_to_npz(in_path, out_path, sim, scene)
    else:
        print("ERROR: Provide --input/--output for single file, or --input_dir/--output_dir for batch mode.")


if __name__ == "__main__":
    main()
    simulation_app.close()
