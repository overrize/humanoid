from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_orientation_l2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)

    num_feet = len(sensor_cfg.body_ids)

    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]  # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)

    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def ang_vel_xy(
    env: ManagerBasedRLEnv, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3
    return torch.exp(torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1) * -2.0) * base_height


def lin_vel_xy(
    env: ManagerBasedRLEnv, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3
    return torch.exp(torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * -5.0) * base_height


def target_orientation(
    env: ManagerBasedRLEnv, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    standup = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3
    return torch.exp(torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * -5) * standup


def target_base_height(
    env: ManagerBasedRLEnv, base_height_target: float, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    standup = base_height > target_base_height_phase3
    return torch.exp(torch.abs(base_height - base_height_target) * -20.0) * standup


def target_joint_deviation_l2(
    env: ManagerBasedRLEnv, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    standup = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3
    return torch.sum(torch.square(angle), dim=1) * standup
