import os

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.deepmimic.deepmimic_env_cfg import DeepMimicEnvCfg

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]
ANIMATION_TERM_NAME = "animation"


@configclass
class G1DeepMimicEnvCfg(DeepMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "deepmimic_v2"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            # Walk / locomotion (weight 1.0)
            "B10_-__Walk_turn_left_45_stageii": 1.0,
            "B11_-__Walk_turn_left_135_stageii": 1.0,
            "B13_-__Walk_turn_right_90_stageii": 1.0,
            "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            "B15_-__Walk_turn_around_stageii": 1.0,
            "B22_-__side_step_left_stageii": 1.0,
            "B23_-__side_step_right_stageii": 1.0,
            "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
            "B9_-__Walk_turn_left_90_stageii": 1.0,
            "Walk_B10_-_Walk_turn_left_45_stageii": 1.0,
            "Walk_B13_-_Walk_turn_right_45_stageii": 1.0,
            "Walk_B15_-_Walk_turn_around_stageii": 1.0,
            "Walk_B16_-_Walk_turn_change_stageii": 1.0,
            "Walk_B22_-_Side_step_left_stageii": 1.0,
            "Walk_B23_-_Side_step_right_stageii": 1.0,
            "Walk_B4_-_Stand_to_Walk_Back_stageii": 1.0,
            # Run / agile locomotion (weight 1.0)
            "C1_-_stand_to_run_stageii": 1.0,
            "C3_-_run_stageii": 1.0,
            "C4_-_run_to_walk_a_stageii": 1.0,
            "C5_-_walk_to_run_stageii": 1.0,
            "C6_-_stand_to_run_backwards_stageii": 1.0,
            "C8_-_run_backwards_to_stand_stageii": 1.0,
            "C9_-_run_backwards_turn_run_forward_stageii": 1.0,
            "C11_-_run_turn_left_90_stageii": 1.0,
            "C12_-_run_turn_left_45_stageii": 1.0,
            "C13_-_run_turn_left_135_stageii": 1.0,
            "C14_-_run_turn_right_90_stageii": 1.0,
            "C15_-_run_turn_right_45_stageii": 1.0,
            "C16_-_run_turn_right_135_stageii": 1.0,
            "C17_-_run_change_direction_stageii": 1.0,
            # Acrobatic transitions (upweight — fewer clips)
            "C18_-_run_to_hop_to_walk_stageii": 2.0,
            "C26_-_run_to_crouch_stageii": 2.0,
            # Martial arts / evasion (upweight — rarest category)
            "E5_-__hook_left_stageii": 2.0,
            "G5_-__back_kick_stageii": 2.0,
            "G10-__roundhouse_leading_left_stageii": 2.0,
            "G12-__cresent_left_stageii": 2.0,
            "G13-__cresent_right_stageii": 2.0,
            "G19-__reverse_spin_cresent_left_stageii": 2.0,
            "G20_-__reverse_spin_cresent_right_stageii": 2.0,
        }

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.policy.ref_root_pos_error.params = {"animation": ANIMATION_TERM_NAME}
        self.observations.policy.ref_root_rot_tan_norm.params = {"animation": ANIMATION_TERM_NAME}
        self.observations.policy.ref_joint_pos.params = {"animation": ANIMATION_TERM_NAME}
        self.observations.policy.ref_key_body_pos_b.params = {"animation": ANIMATION_TERM_NAME}

        # -----------------------------------------------------
        # Events
        # -----------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com.params["asset_cfg"].body_names = "torso_link"
        self.events.reset_from_ref.params = {"animation": ANIMATION_TERM_NAME, "height_offset": 0.1}
        # self.events.reset_from_ref = None

        # -----------------------------------------------------
        # Rewards
        # -----------------------------------------------------
        self.rewards.ref_track_root_pos_w_error_exp.weight = 0.15
        self.rewards.ref_track_root_pos_w_error_exp.params = {
            "std": 0.5,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_quat_error_exp.weight = 0.08
        self.rewards.ref_track_quat_error_exp.params = {
            "std": 0.5,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_root_vel_w_error_exp.weight = 0.1
        self.rewards.ref_track_root_vel_w_error_exp.params = {
            "std": 1.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_root_ang_vel_w_error_exp.weight = 0.05
        self.rewards.ref_track_root_ang_vel_w_error_exp.params = {
            "std": 1.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_key_body_pos_b_error_exp.weight = 0.15
        self.rewards.ref_track_key_body_pos_b_error_exp.params = {
            "std": 0.3,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True),
        }
        self.rewards.ref_track_dof_pos_error_exp.weight = 0.5
        self.rewards.ref_track_dof_pos_error_exp.params = {
            "std": 2.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_dof_vel_error_exp.weight = 0.1
        self.rewards.ref_track_dof_vel_error_exp.params = {
            "std": 10.0,
            "animation": ANIMATION_TERM_NAME,
        }

        # -----------------------------------------------------
        # Terminations
        # -----------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "waist_yaw_link",
            "pelvis",
            ".*_shoulder_.*_link",
            ".*_elbow_link",
        ]
        self.terminations.deviation_root_pos_w.params = {
            "threshold": 2.0,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.deviation_key_body_pos_w.params = {
            "threshold": 2.0,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True),
        }

        self.terminations.bad_orientation = None


# For debug only
@configclass
class G1DeepMimicEnvCfg_DEBUG(G1DeepMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 8
        self.scene.env_spacing = 3.0

        self.scene.robot_anim = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_anim")
        self.scene.robot_anim.spawn.rigid_props.disable_gravity = True  # type: ignore
        self.scene.robot_anim.spawn.articulation_props.enabled_self_collisions = False  # type: ignore
        self.scene.robot_anim.spawn.activate_contact_sensors = False  # type: ignore
        self.scene.robot_anim.spawn.collision_props = sim_utils.CollisionPropertiesCfg(  # type: ignore
            collision_enabled=False
        )

        self.animation.animation.enable_visualization = True
        self.animation.animation.vis_root_offset = [2.0, 0.0, 0.0]
        self.animation.animation.random_initialize = False

        # self.terminations.bad_orientation = None
        # self.terminations.base_height = None
        # self.terminations.base_contact = None


@configclass
class G1DeepMimicEnvCfg_PLAY(G1DeepMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.animation.animation.random_initialize = False
