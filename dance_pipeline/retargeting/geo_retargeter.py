"""
Geometric retargeter: joint angles from 3-D landmark positions.

No IK, no SMPL. Works directly with MediaPipe or BVH NSF output.

Approach per limb:
  - Build a local reference frame from trunk/pelvis landmarks
  - Express the distal bone vector in that frame
  - Map spherical coordinates → robot DOF axes
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .base import Retargeter
from ..nsf.format import NSFSequence, Joint

# ─── small helpers ────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    c = np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0)
    return float(np.arccos(c))


def _rot_frame(right: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix [right, up, forward] as columns."""
    right = _unit(right)
    fwd   = _unit(np.cross(right, up))
    up    = np.cross(fwd, right)
    return np.column_stack([right, up, fwd])   # world→local


def _in_frame(vec: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Express world vector in local frame (frame columns = local axes in world)."""
    return frame.T @ vec


# ─── hip/shoulder decomposition ──────────────────────────────────────────────

def _limb_to_ypr(bone_local: np.ndarray) -> tuple[float, float, float]:
    """
    Given a bone direction expressed in a local torso frame (y-up, x-right,
    z-forward), return (pitch, roll, yaw) in robot joint convention:
      pitch = rotation around y (fore-aft swing)
      roll  = rotation around x (abduction/adduction)
      yaw   = rotation around z (internal/external, from cross product sign)
    """
    x, y, z = bone_local
    # elevation above horizontal
    pitch = float(np.arcsin(np.clip(-y, -1, 1)))
    # azimuth in xz plane
    roll  = float(np.arctan2(-x, -z))   # abduction = lateral
    yaw   = 0.0                           # cannot recover twist from positions alone
    return pitch, roll, yaw


# ─── main retargeter ─────────────────────────────────────────────────────────

class GeoRetargeter(Retargeter):
    """
    Geometric retargeter: computes G1 DOF angles analytically from
    NSF 3-D joint positions. Compatible with MediaPipe and BVH extractors.

    Joint twist (yaw for hips, yaw for shoulders, wrist yaw) is zeroed
    because it cannot be recovered from landmark positions alone — these DOFs
    require rotation matrices (use SMPLRetargeter when hmr2 is available).
    """

    _SMOOTH_SIGMA = 1.5   # Gaussian smoothing over time (frames)

    DOF_NAMES = [
        "left_hip_pitch",  "left_hip_roll",  "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch",  "left_shoulder_roll",  "left_shoulder_yaw",
        "left_elbow",
        "left_wrist_roll",  "left_wrist_pitch",  "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    ]

    _LIMITS = {
        "left_hip_pitch":    (-1.57, 2.53),
        "left_hip_roll":     (-0.52, 2.53),
        "left_hip_yaw":      (-0.87, 0.87),
        "left_knee":         (-0.09, 2.79),
        "left_ankle_pitch":  (-0.87, 0.52),
        "left_ankle_roll":   (-0.35, 0.35),
        "right_hip_pitch":   (-1.57, 2.53),
        "right_hip_roll":    (-2.53, 0.52),
        "right_hip_yaw":     (-0.87, 0.87),
        "right_knee":        (-0.09, 2.79),
        "right_ankle_pitch": (-0.87, 0.52),
        "right_ankle_roll":  (-0.35, 0.35),
        "waist_yaw":         (-2.62, 2.62),
        "waist_roll":        (-0.52, 0.52),
        "waist_pitch":       (-0.87, 0.87),
        "left_shoulder_pitch":  (-3.14, 2.79),
        "left_shoulder_roll":   (-1.57, 2.79),
        "left_shoulder_yaw":    (-1.57, 4.71),
        "left_elbow":           (-1.57, 2.09),
        "left_wrist_roll":      (-1.57, 1.57),
        "left_wrist_pitch":     (-1.57, 1.57),
        "left_wrist_yaw":       (-1.57, 1.57),
        "right_shoulder_pitch": (-3.14, 2.79),
        "right_shoulder_roll":  (-2.79, 1.57),
        "right_shoulder_yaw":   (-4.71, 1.57),
        "right_elbow":          (-1.57, 2.09),
        "right_wrist_roll":     (-1.57, 1.57),
        "right_wrist_pitch":    (-1.57, 1.57),
        "right_wrist_yaw":      (-1.57, 1.57),
    }

    @property
    def robot_name(self) -> str:
        return "g1"

    def retarget(self, seq: NSFSequence) -> dict:
        pos = seq.positions   # (T, 23, 3) y-up
        T   = len(pos)
        dof = np.zeros((T, len(self.DOF_NAMES)), dtype=np.float32)

        for t in range(T):
            p = pos[t]   # (23, 3)  indexed by Joint enum

            # ── Pelvis frame ─────────────────────────────────────────────────
            # right = r_hip → l_hip direction (robot: left = +x)
            pel_right = _unit(p[Joint.L_HIP] - p[Joint.R_HIP])
            pel_up    = np.array([0.0, 1.0, 0.0])
            pel_frame = _rot_frame(pel_right, pel_up)  # (3,3) columns=right,up,fwd

            # ── Chest frame (for shoulders) ──────────────────────────────────
            chest_right = _unit(p[Joint.L_SHOULDER] - p[Joint.R_SHOULDER])
            chest_up    = _unit(p[Joint.NECK] - p[Joint.CHEST])
            chest_frame = _rot_frame(chest_right, chest_up)

            col = 0

            # ── LEFT HIP ─────────────────────────────────────────────────────
            thigh_L = _unit(p[Joint.L_KNEE] - p[Joint.L_HIP])
            thigh_L_loc = _in_frame(thigh_L, pel_frame)
            pitch, roll, _ = _limb_to_ypr(thigh_L_loc)
            dof[t, col:col+3] = [pitch, roll, 0.0]; col += 3

            # ── LEFT KNEE (flexion angle) ─────────────────────────────────────
            # straight leg → thigh and shin parallel → angle ≈ 0; bent → increases
            shin_L  = _unit(p[Joint.L_ANKLE] - p[Joint.L_KNEE])
            dof[t, col] = _angle_between(thigh_L, shin_L); col += 1

            # ── LEFT ANKLE ───────────────────────────────────────────────────
            foot_L = _unit(p[Joint.L_FOOT] - p[Joint.L_ANKLE])
            foot_L_loc = _in_frame(foot_L, pel_frame)
            dof[t, col]   = float(np.arctan2(-foot_L_loc[1], -foot_L_loc[2]))  # pitch
            dof[t, col+1] = float(np.arctan2(-foot_L_loc[0], -foot_L_loc[2]))  # roll
            col += 2

            # ── RIGHT HIP ────────────────────────────────────────────────────
            thigh_R = _unit(p[Joint.R_KNEE] - p[Joint.R_HIP])
            thigh_R_loc = _in_frame(thigh_R, pel_frame)
            pitch, roll, _ = _limb_to_ypr(thigh_R_loc)
            dof[t, col:col+3] = [pitch, -roll, 0.0]; col += 3   # roll sign flipped for right side

            # ── RIGHT KNEE ───────────────────────────────────────────────────
            shin_R = _unit(p[Joint.R_ANKLE] - p[Joint.R_KNEE])
            dof[t, col] = _angle_between(thigh_R, shin_R); col += 1

            # ── RIGHT ANKLE ──────────────────────────────────────────────────
            foot_R = _unit(p[Joint.R_FOOT] - p[Joint.R_ANKLE])
            foot_R_loc = _in_frame(foot_R, pel_frame)
            dof[t, col]   = float(np.arctan2(-foot_R_loc[1], -foot_R_loc[2]))
            dof[t, col+1] = float(np.arctan2( foot_R_loc[0], -foot_R_loc[2]))
            col += 2

            # ── WAIST (chest vs pelvis relative orientation) ─────────────────
            # yaw: rotation around vertical axis
            pel_fwd   = pel_frame[:, 2]
            chest_fwd = chest_frame[:, 2]
            waist_yaw   = float(np.arctan2(
                np.dot(chest_fwd, pel_frame[:, 0]),
                np.dot(chest_fwd, pel_frame[:, 2]),
            ))
            waist_roll  = float(np.arctan2(
                p[Joint.CHEST][0] - p[Joint.ROOT][0],
                p[Joint.CHEST][1] - p[Joint.ROOT][1],
            ))
            chest_up_local = _in_frame(chest_frame[:, 1], pel_frame)
            waist_pitch = float(np.arctan2(-chest_up_local[2], chest_up_local[1]))
            dof[t, col:col+3] = [waist_yaw, waist_roll, waist_pitch]; col += 3

            # ── LEFT SHOULDER ─────────────────────────────────────────────────
            uarm_L = _unit(p[Joint.L_ELBOW] - p[Joint.L_SHOULDER])
            uarm_L_loc = _in_frame(uarm_L, chest_frame)
            pitch, roll, _ = _limb_to_ypr(uarm_L_loc)
            dof[t, col:col+3] = [pitch, roll, 0.0]; col += 3

            # ── LEFT ELBOW ────────────────────────────────────────────────────
            farm_L = _unit(p[Joint.L_WRIST] - p[Joint.L_ELBOW])
            dof[t, col] = _angle_between(uarm_L, farm_L); col += 1

            # ── LEFT WRIST (zeroed — needs rotation matrices) ─────────────────
            col += 3

            # ── RIGHT SHOULDER ────────────────────────────────────────────────
            uarm_R = _unit(p[Joint.R_ELBOW] - p[Joint.R_SHOULDER])
            uarm_R_loc = _in_frame(uarm_R, chest_frame)
            pitch, roll, _ = _limb_to_ypr(uarm_R_loc)
            dof[t, col:col+3] = [pitch, -roll, 0.0]; col += 3

            # ── RIGHT ELBOW ───────────────────────────────────────────────────
            farm_R = _unit(p[Joint.R_WRIST] - p[Joint.R_ELBOW])
            dof[t, col] = _angle_between(uarm_R, farm_R); col += 1

            # ── RIGHT WRIST (zeroed) ──────────────────────────────────────────
            col += 3

        # Smooth over time
        dof = gaussian_filter1d(dof, sigma=self._SMOOTH_SIGMA, axis=0)

        # Clip to joint limits
        for i, name in enumerate(self.DOF_NAMES):
            lo, hi = self._LIMITS[name]
            dof[:, i] = np.clip(dof[:, i], lo, hi)

        # Velocities
        dt = 1.0 / seq.fps
        vel = np.gradient(dof, dt, axis=0).astype(np.float32)

        # Root pose from pelvis
        root_positions = pos[:, int(Joint.ROOT), :].astype(np.float32)
        pel_rights = _unit(pos[:, int(Joint.L_HIP), :] - pos[:, int(Joint.R_HIP), :])
        yaws = np.arctan2(pel_rights[:, 2], pel_rights[:, 0])
        half = yaws / 2.0
        root_rotations = np.stack([
            np.cos(half), np.zeros_like(half), np.sin(half), np.zeros_like(half)
        ], axis=-1).astype(np.float32)  # wxyz, yaw only

        return {
            "root_positions": root_positions,
            "root_rotations": root_rotations,
            "dof_positions":  dof.astype(np.float32),
            "dof_velocities": vel,
            "dof_names":      self.DOF_NAMES,
            "fps":            seq.fps,
        }
