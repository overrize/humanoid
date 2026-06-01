import os
from dataclasses import MISSING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG

KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 10
BASE_BODY_NAME = "torso_link"
TARGET_BASE_HEIGHT_PHASE3 = 0.65
BASE_HEIGHT_TARGET = 0.75


@configclass
class CommandsCfg:
    force_command = mdp.ForceCommandCfg(
        force=200.0,
        resampling_time_range=[100.0, 100.0],
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
            noise=Unoise(n_min=-0.08, n_max=0.08),
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        root_height = ObsTerm(func=mdp.base_pos_z)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class DiscriminatorCfg(ObsGroup):
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 10
            self.flatten_history_dim = False

    disc: DiscriminatorCfg = DiscriminatorCfg()

    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        ref_root_local_rot_tan_norm = ObsTerm(
            func=mdp.ref_root_local_rot_tan_norm,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            },
        )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            },
        )
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            },
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            },
        )
        ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    apply_force = EventTerm(
        func=mdp.apply_force,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_BODY_NAME]),
        },
    )

    reset_from_ref = EventTerm(
        func=mdp.reset_from_ref,
        mode="reset",
        params=MISSING,
    )


@configclass
class RewardsCfg:
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.005,
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy,
        weight=2,
        params={
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    lin_vel_xy = RewTerm(
        func=mdp.lin_vel_xy,
        weight=2,
        params={
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    target_orientation = RewTerm(
        func=mdp.target_orientation,
        weight=2,
        params={
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    target_base_height = RewTerm(
        func=mdp.target_base_height,
        weight=5.0,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    target_joint_deviation_l2 = RewTerm(
        func=mdp.target_joint_deviation_l2,
        weight=-0.1,
        params={
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    force_level = CurrTerm(
        func=mdp.force_level,
        params={
            "reward_term_name": "target_base_height",
        },
    )


@configclass
class G1AmpEnvCfg(LocomotionAmpEnvCfg):
    observations = ObservationsCfg()
    commands = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "get_up"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "fallAndGetUp1_subject1_680_800": 0.0,
            "fallAndGetUp1_subject1_850_940": 0.0,
            "fallAndGetUp1_subject1_1060_1150": 1.0,
            "fallAndGetUp1_subject1_1400_1480": 1.0,
            "fallAndGetUp1_subject1_1600_1700": 0.0,
            "fallAndGetUp1_subject1_2100_2200": 1.0,
            "fallAndGetUp1_subject1_2300_2400": 0.0,
            "fallAndGetUp1_subject4_3700_3800": 0.0,
            "fallAndGetUp1_subject5_2500_2600": 1.0,
            "fallAndGetUp2_subject2_360_580": 0.0,
            "fallAndGetUp2_subject2_850_1050": 1.0,
            "fallAndGetUp2_subject2_1200_1370": 0.0,
            "fallAndGetUp2_subject2_1500_1600": 0.0,
            "fallAndGetUp2_subject3_900_1000": 1.0,
            "fallAndGetUp2_subject3_1850_1920": 0.0,
            "fallAndGetUp2_subject3_2080_2180": 0.0,
            "fallAndGetUp6_subject1_530_600": 1.0,
            "fallAndGetUp6_subject1_650_700": 1.0,
            "fallAndGetUp6_subject1_1080_1180": 1.0,
            "fallAndGetUp6_subject1_1230_1300": 0.0,
            "fallAndGetUp6_subject1_1630_1690": 1.0,
        }

        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }

        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }

        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        self.events.add_base_mass.params["asset_cfg"].body_names = BASE_BODY_NAME
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1,
        }


@configclass
class G1AmpEnvCfg_PLAY(G1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.force_command.force = 0.0

        self.animation.animation.random_initialize = False
        self.animation.animation.random_fetch = False
