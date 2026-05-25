"""
SMPL-to-G1 retargeter: maps HMR2 rotation matrices directly to G1 DOF angles.

No IK. Each SMPL joint rotation matrix is decomposed into the corresponding
G1 DOF axes. For single-DOF joints the rotation projected onto the joint axis
is extracted; for multi-DOF joints the rotation is decomposed into the kinematic
chain order (pitch→roll→yaw or similar).

SMPL body_pose indices (0-indexed, after global_orient):
  0=l_hip  1=r_hip  2=spine1  3=l_knee  4=r_knee  5=spine2
  6=l_ankle  7=r_ankle  8=spine3  9=l_foot 10=r_foot 11=neck
 12=l_collar 13=r_collar 14=head 15=l_shoulder 16=r_shoulder
 17=l_elbow 18=r_elbow 19=l_wrist 20=r_wrist 21=l_hand 22=r_hand
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .base import Retargeter as BaseRetargeter
from ..nsf.format import NSFSequence


# ─── axis helpers ────────────────────────────────────────────────────────────

_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def _project_angle(R: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle of rotation R projected onto `axis` (radians)."""
    rotvec = Rotation.from_matrix(R).as_rotvec()
    # project rotation vector onto axis; magnitude with sign
    proj = np.dot(rotvec, axis / np.linalg.norm(axis))
    return float(proj)


def _decompose_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose R = Ry(pitch) @ Rx(roll) @ Rz(yaw).
    Returns (pitch, roll, yaw) in radians.
    Used for: hip (pitch→roll→yaw), shoulder (pitch→roll→yaw).
    """
    # R[1,2] = -sin(roll)
    roll  = float(np.arcsin(-np.clip(R[1, 2], -1, 1)))
    cr    = np.cos(roll)
    if abs(cr) > 1e-6:
        pitch = float(np.arctan2(R[0, 2], R[2, 2]))
        yaw   = float(np.arctan2(R[1, 0], R[1, 1]))
    else:                          # gimbal lock: pitch absorbs all
        pitch = float(np.arctan2(-R[2, 0], R[0, 0]))
        yaw   = 0.0
    return pitch, roll, yaw


def _decompose_zxy(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose R = Rz(yaw) @ Rx(roll) @ Ry(pitch).
    Returns (yaw, roll, pitch).
    Used for: waist (yaw→roll→pitch).
    """
    roll  = float(np.arcsin(np.clip(R[2, 1], -1, 1)))
    cr    = np.cos(roll)
    if abs(cr) > 1e-6:
        yaw   = float(np.arctan2(-R[0, 1], R[1, 1]))
        pitch = float(np.arctan2(-R[2, 0], R[2, 2]))
    else:
        yaw   = float(np.arctan2(R[1, 0], R[0, 0]))
        pitch = 0.0
    return yaw, roll, pitch


def _decompose_xyx(R: np.ndarray, axis: np.ndarray) -> float:
    """Project onto a single axis — for 1-DOF joints."""
    return _project_angle(R, axis)


# ─── spine composition ───────────────────────────────────────────────────────

def _compose_spine(R1: np.ndarray, R2: np.ndarray, R3: np.ndarray) -> np.ndarray:
    """Compose three spine segment rotations (spine1, spine2, spine3)."""
    return Rotation.from_matrix(R1) * Rotation.from_matrix(R2) * Rotation.from_matrix(R3)


# ─── SMPL → G1 joint indices ─────────────────────────────────────────────────
# body_pose index (0=l_hip, after global_orient)
_BP = {
    "l_hip":       0,   "r_hip":      1,
    "spine1":      2,
    "l_knee":      3,   "r_knee":     4,
    "spine2":      5,
    "l_ankle":     6,   "r_ankle":    7,
    "spine3":      8,
    "l_shoulder":  15,  "r_shoulder": 16,
    "l_elbow":     17,  "r_elbow":    18,
    "l_wrist":     19,  "r_wrist":    20,
}


class SMPLRetargeter(BaseRetargeter):
    """
    Converts an NSFSequence with attached SMPL rotation matrices
    (_smpl_rotmats, shape T×24×3×3) to G1 joint angles.

    G1 DOF order (matches legged_lab MotionLoader default):
      left_hip_pitch, left_hip_roll, left_hip_yaw,
      left_knee,
      left_ankle_pitch, left_ankle_roll,
      right_hip_pitch, right_hip_roll, right_hip_yaw,
      right_knee,
      right_ankle_pitch, right_ankle_roll,
      waist_yaw, waist_roll, waist_pitch,
      left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
      left_elbow,
      left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
      right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
      right_elbow,
      right_wrist_roll, right_wrist_pitch, right_wrist_yaw
    """

    @property
    def robot_name(self) -> str:
        return "g1"

    # joint limits from g1_29dof URDF (rad), approximately
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

    DOF_NAMES = list(_LIMITS.keys())

    def retarget(self, seq: NSFSequence) -> dict:
        """
        Parameters
        ----------
        seq : NSFSequence with seq._smpl_rotmats (T,24,3,3)

        Returns
        -------
        dict compatible with motion_builder.build_npz
        """
        if not hasattr(seq, "_smpl_rotmats"):
            raise ValueError(
                "NSFSequence has no _smpl_rotmats. "
                "Use SMPLExtractor (not MediaPipeExtractor) to produce the sequence."
            )

        rm = seq._smpl_rotmats   # (T,24,3,3);  rm[:,0]=global_orient, rm[:,1:]=body_pose
        bp = rm[:, 1:, :, :]    # (T,23,3,3)   body_pose

        T = len(rm)
        dof = np.zeros((T, 29), dtype=np.float32)

        for t in range(T):
            col = 0

            # ── LEFT HIP (pitch, roll, yaw) ──────────────────────────────
            pitch, roll, yaw = _decompose_yxz(bp[t, _BP["l_hip"]])
            dof[t, col:col+3] = [pitch, roll, yaw]; col += 3

            # ── LEFT KNEE (y-axis pitch only) ────────────────────────────
            dof[t, col] = _project_angle(bp[t, _BP["l_knee"]], _Y); col += 1

            # ── LEFT ANKLE (pitch, roll) ──────────────────────────────────
            Ra = bp[t, _BP["l_ankle"]]
            dof[t, col]   = _project_angle(Ra, _Y)   # pitch
            dof[t, col+1] = _project_angle(Ra, _X)   # roll
            col += 2

            # ── RIGHT HIP (pitch, roll, yaw) ──────────────────────────────
            pitch, roll, yaw = _decompose_yxz(bp[t, _BP["r_hip"]])
            dof[t, col:col+3] = [pitch, roll, yaw]; col += 3

            # ── RIGHT KNEE ────────────────────────────────────────────────
            dof[t, col] = _project_angle(bp[t, _BP["r_knee"]], _Y); col += 1

            # ── RIGHT ANKLE ───────────────────────────────────────────────
            Ra = bp[t, _BP["r_ankle"]]
            dof[t, col]   = _project_angle(Ra, _Y)
            dof[t, col+1] = _project_angle(Ra, _X)
            col += 2

            # ── WAIST (yaw, roll, pitch — composed from 3 spine segs) ────
            R_spine = _compose_spine(
                bp[t, _BP["spine1"]],
                bp[t, _BP["spine2"]],
                bp[t, _BP["spine3"]],
            ).as_matrix()
            yaw_w, roll_w, pitch_w = _decompose_zxy(R_spine)
            dof[t, col:col+3] = [yaw_w, roll_w, pitch_w]; col += 3

            # ── LEFT SHOULDER (pitch, roll, yaw) ─────────────────────────
            pitch, roll, yaw = _decompose_yxz(bp[t, _BP["l_shoulder"]])
            dof[t, col:col+3] = [pitch, roll, yaw]; col += 3

            # ── LEFT ELBOW ────────────────────────────────────────────────
            dof[t, col] = _project_angle(bp[t, _BP["l_elbow"]], _Y); col += 1

            # ── LEFT WRIST (roll, pitch, yaw) ─────────────────────────────
            Rw = bp[t, _BP["l_wrist"]]
            dof[t, col]   = _project_angle(Rw, _X)   # roll
            dof[t, col+1] = _project_angle(Rw, _Y)   # pitch
            dof[t, col+2] = _project_angle(Rw, _Z)   # yaw
            col += 3

            # ── RIGHT SHOULDER (pitch, roll, yaw) ─────────────────────────
            pitch, roll, yaw = _decompose_yxz(bp[t, _BP["r_shoulder"]])
            dof[t, col:col+3] = [pitch, roll, yaw]; col += 3

            # ── RIGHT ELBOW ───────────────────────────────────────────────
            dof[t, col] = _project_angle(bp[t, _BP["r_elbow"]], _Y); col += 1

            # ── RIGHT WRIST (roll, pitch, yaw) ────────────────────────────
            Rw = bp[t, _BP["r_wrist"]]
            dof[t, col]   = _project_angle(Rw, _X)
            dof[t, col+1] = _project_angle(Rw, _Y)
            dof[t, col+2] = _project_angle(Rw, _Z)
            col += 3

        # Clip to joint limits
        for i, name in enumerate(self.DOF_NAMES):
            lo, hi = self._LIMITS[name]
            dof[:, i] = np.clip(dof[:, i], lo, hi)

        # Velocities via central differences
        dt = 1.0 / seq.fps
        vel = np.gradient(dof, dt, axis=0).astype(np.float32)

        # Root pose from global_orient (SMPL joint 0)
        global_orient = rm[:, 0, :, :]  # (T,3,3)
        root_quats = Rotation.from_matrix(global_orient).as_quat()  # xyzw
        root_quats_wxyz = np.concatenate([root_quats[:, 3:4], root_quats[:, :3]], axis=-1)

        root_positions = seq.positions[:, 0, :] if seq.positions is not None else np.zeros((T, 3))

        return {
            "root_positions":  root_positions.astype(np.float32),
            "root_rotations":  root_quats_wxyz.astype(np.float32),
            "dof_positions":   dof,
            "dof_velocities":  vel,
            "dof_names":       self.DOF_NAMES,
            "fps":             seq.fps,
        }
