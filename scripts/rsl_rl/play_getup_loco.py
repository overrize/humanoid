"""Dual-policy play: AMP get-up (fallen) → WBT stand/locomotion (standing).

The AMP env starts robots in fallen poses. When root height exceeds the
threshold the WBT actor takes over with a "hold default pose" reference,
which makes the robot stabilise upright.

Usage (single line):
    conda run -n env_isaaclab python scripts/rsl_rl/play_getup_loco.py --amp_checkpoint logs/rsl_rl/g1_amp_get_up/2026-06-02_13-55-01/model_52500.pt --wbt_checkpoint models/wbt_g1_v1/model_29999.pt --num_envs 48
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--amp_task", type=str, default="LeggedLab-Isaac-AMP-G1-GET-UP-Play-v0")
parser.add_argument("--amp_checkpoint", type=str, required=True)
parser.add_argument("--wbt_checkpoint", type=str, required=True)
parser.add_argument("--height_threshold", type=float, default=0.5,
                    help="Root height (m) above which WBT actor is active")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import legged_lab.tasks  # noqa: F401  registers AMP tasks


# ---------------------------------------------------------------------------
# Shared actor network (ELU activations, no normaliser)
# ---------------------------------------------------------------------------
class SimpleActor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: list):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _load_actor(path: str, in_dim: int, device: str) -> SimpleActor:
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    out_dim = sd["actor.6.weight"].shape[0]
    actor = SimpleActor(in_dim, out_dim, [512, 256, 128])
    actor.load_state_dict({"net." + k[len("actor."):]: v
                           for k, v in sd.items() if k.startswith("actor.")})
    actor.eval()
    return actor.to(device)


# ---------------------------------------------------------------------------
# WBT obs (160 dims) with "hold default pose" reference
#
# Layout (must match Tracking-Flat-G1-v0 policy obs group):
#   command         : ref_jpos(29) + ref_jvel(29)          = 58
#   anchor_pos_b    : [0,0,0]                               =  3
#   anchor_ori_b    : identity rel-ori (2 cols of 3x3)      =  6
#   base_lin_vel    : actual                                =  3
#   base_ang_vel    : actual                                =  3
#   joint_pos_rel   : jpos - default_jpos                  = 29
#   joint_vel_rel   : jvel - default_jvel                  = 29
#   actions         : last actions                         = 29
#                                                    total = 160
# ---------------------------------------------------------------------------
def compute_wbt_obs(robot, last_actions: torch.Tensor) -> torch.Tensor:
    N = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device

    ref_jpos = robot.data.default_joint_pos                          # (N,29)
    ref_jvel = torch.zeros_like(ref_jpos)                            # (N,29)
    command = torch.cat([ref_jpos, ref_jvel], dim=-1)                # (N,58)

    anchor_pos_b = torch.zeros(N, 3, device=device)                  # (N,3)

    # Identity relative orientation: first 2 cols of 3x3 identity → [1,0,0, 0,1,0]
    anchor_ori_b = torch.zeros(N, 6, device=device)
    anchor_ori_b[:, 0] = 1.0   # col-0 x
    anchor_ori_b[:, 4] = 1.0   # col-1 y

    lin_vel = robot.data.root_lin_vel_b                              # (N,3)
    ang_vel = robot.data.root_ang_vel_b                              # (N,3)

    jpos_rel = robot.data.joint_pos - robot.data.default_joint_pos  # (N,29)
    jvel_rel = robot.data.joint_vel - robot.data.default_joint_vel  # (N,29)

    return torch.cat([command, anchor_pos_b, anchor_ori_b,
                      lin_vel, ang_vel, jpos_rel, jvel_rel, last_actions], dim=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra_task_config(args_cli.amp_task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    device = env_cfg.sim.device

    # Disable policy obs noise for clean playback
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.amp_task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    num_envs = env.num_envs

    robot = env.unwrapped.scene["robot"]

    amp_actor = _load_actor(os.path.abspath(args_cli.amp_checkpoint), 570, device)
    wbt_actor = _load_actor(os.path.abspath(args_cli.wbt_checkpoint), 160, device)
    print(f"[play] AMP actor loaded (570→29)  |  WBT actor loaded (160→29)")
    print(f"[play] height_threshold = {args_cli.height_threshold} m  |  envs = {num_envs}")

    last_actions = torch.zeros(num_envs, 29, device=device)

    # Initial obs
    obs_td = env.get_observations()

    while simulation_app.is_running():
        with torch.inference_mode():
            root_height = robot.data.root_link_pos_w[:, 2]   # (N,)
            fallen = root_height < args_cli.height_threshold  # (N,) bool

            amp_obs = obs_td["policy"]                        # (N, 570) from env
            amp_actions = amp_actor(amp_obs)                  # (N, 29)

            wbt_obs = compute_wbt_obs(robot, last_actions)    # (N, 160)
            wbt_actions = wbt_actor(wbt_obs)                  # (N, 29)

            actions = torch.where(fallen.unsqueeze(-1), amp_actions, wbt_actions)

            obs_td, _, _, _ = env.step(actions)
            last_actions = actions.clone()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
