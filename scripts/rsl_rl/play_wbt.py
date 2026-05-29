"""Play a trained BeyondMimic whole-body tracking policy from a local checkpoint.

Usage:
    conda run -n env_isaaclab python scripts/rsl_rl/play_wbt.py \\
        --task Tracking-Flat-G1-v0 \\
        --motion_file /tmp/combined_g1_50fps.npz \\
        --load_run 2026-05-28_20-33-49 \\
        --num_envs 1
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a BeyondMimic G1 whole-body tracking policy.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    motion_file = os.path.abspath(args_cli.motion_file)
    assert os.path.isfile(motion_file), f"motion_file not found: {motion_file}"
    env_cfg.commands.motion.motion_file = motion_file
    print(f"[play_wbt] motion_file: {motion_file}")

    # Disable noise for clean playback
    env_cfg.observations.policy.enable_corruption = False

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[play_wbt] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        log_dir = os.path.dirname(resume_path)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            ret = env.step(actions)
            obs = ret[0]
        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
