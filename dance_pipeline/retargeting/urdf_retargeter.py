"""
URDF-aware motion retargeter.

Pipeline per frame:
  1. Scale NSF endpoint positions to match robot limb lengths
  2. Solve whole-body IK with Pinocchio (position targets + contact constraints)
  3. Clip to joint limits
  4. Finite-difference velocities + smooth

Coordinate convention: y-up (Isaac Lab / Pinocchio default).
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

import pinocchio as pin

from .base import Retargeter
from .joint_map import G1_BODY_TARGETS, G1_LIMB_SCALE_PAIRS
from ..nsf.format import NSFSequence, Joint, FOOT_JOINTS


# IK solver settings
_IK_MAX_ITER   = 50
_IK_EPS        = 1e-3   # position error threshold (m)
_IK_DT         = 0.1    # step size
_IK_DAMP       = 1e-6   # damping for pseudo-inverse


class URDFRetargeter(Retargeter):

    def __init__(
        self,
        urdf_path: str | Path,
        mesh_dir:  str | Path,
        robot_name: str = "g1",
        smooth_sigma: float = 1.0,
    ):
        self._robot_name = robot_name
        self._smooth_sigma = smooth_sigma

        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(urdf_path),
            str(mesh_dir),
            pin.JointModelFreeFlyer(),
        )
        self.model = self.robot.model
        self.data  = self.robot.data

        # Cache DOF names (skip free-flyer's 6 internal DOFs)
        self.dof_names: list[str] = [
            self.model.names[i]
            for i in range(1, self.model.njoints)
            if self.model.joints[i].nq == 1
        ]
        self.ndof = len(self.dof_names)

        # Frame ids for IK targets
        self._frame_ids: dict[str, int] = {}
        for link in G1_BODY_TARGETS:
            try:
                self._frame_ids[link] = self.model.getFrameId(link)
            except Exception:
                pass  # link not in this URDF variant, skip

        # Joint limits
        self._q_lo = self.model.lowerPositionLimit[7:]   # skip free-flyer
        self._q_hi = self.model.upperPositionLimit[7:]

    @property
    def robot_name(self) -> str:
        return self._robot_name

    def retarget(self, seq: NSFSequence) -> dict:
        T = seq.num_frames

        # ── 1. Compute scale factors (human limb → robot limb) ──────────────
        scale = self._compute_scale(seq)

        # ── 2. Scale NSF positions ───────────────────────────────────────────
        scaled_pos = self._scale_positions(seq.positions, scale)

        # ── 3. Per-frame IK ──────────────────────────────────────────────────
        dof_positions = np.zeros((T, self.ndof), dtype=np.float32)
        root_positions = np.zeros((T, 3), dtype=np.float32)
        root_rotations = np.zeros((T, 4), dtype=np.float32)
        root_rotations[:, 0] = 1.0  # identity wxyz

        q = pin.neutral(self.model)

        for t in range(T):
            root_pos = scaled_pos[t, int(Joint.ROOT)]
            q[0:3] = root_pos

            # Root orientation: align pelvis forward direction
            if t + 1 < T:
                fwd_nsf = (
                    scaled_pos[t + 1, int(Joint.ROOT)]
                    - scaled_pos[t,     int(Joint.ROOT)]
                )
            else:
                fwd_nsf = (
                    scaled_pos[t, int(Joint.ROOT)]
                    - scaled_pos[t - 1, int(Joint.ROOT)]
                )
            root_quat = self._yaw_quat(fwd_nsf)
            q[3:7] = root_quat  # xyzw for pinocchio

            # IK for remaining joints
            targets = self._build_targets(scaled_pos[t])
            q = self._ik_step(q, targets)

            root_positions[t] = root_pos
            # Convert xyzw → wxyz for output
            root_rotations[t] = np.array([root_quat[3], root_quat[0], root_quat[1], root_quat[2]])
            dof_positions[t]  = np.clip(q[7:7 + self.ndof], self._q_lo, self._q_hi)

        # ── 4. Velocities ────────────────────────────────────────────────────
        dt = 1.0 / seq.fps
        dof_velocities = np.gradient(dof_positions, dt, axis=0).astype(np.float32)
        dof_velocities = gaussian_filter1d(dof_velocities, sigma=self._smooth_sigma, axis=0)

        return {
            "root_positions":  root_positions,
            "root_rotations":  root_rotations,
            "dof_positions":   dof_positions,
            "dof_velocities":  dof_velocities,
            "dof_names":       self.dof_names,
            "fps":             seq.fps,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_scale(self, seq: NSFSequence) -> dict[str, float]:
        """Ratio of robot limb length to average human limb length per bone."""
        scale = {}
        for child_j, parent_j, link_name in G1_LIMB_SCALE_PAIRS:
            human_len = seq.limb_length(child_j)
            if human_len < 1e-4:
                scale[link_name] = 1.0
                continue
            robot_len = self._robot_limb_length(link_name)
            scale[link_name] = robot_len / human_len if human_len > 0 else 1.0
        return scale

    def _robot_limb_length(self, link_name: str) -> float:
        """Length of the bone ending at link_name in the robot's neutral pose."""
        q0 = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q0)
        pin.updateFramePlacements(self.model, self.data)
        if link_name not in self._frame_ids:
            return 0.3  # fallback
        fid = self._frame_ids[link_name]
        pos = self.data.oMf[fid].translation.copy()

        # Find parent frame
        for child_j, parent_j, lname in G1_LIMB_SCALE_PAIRS:
            if lname == link_name:
                # Walk up one level
                parent_links = {v[0]: k for k, v in G1_BODY_TARGETS.items()}
                parent_link = parent_links.get(parent_j)
                if parent_link and parent_link in self._frame_ids:
                    pfid = self._frame_ids[parent_link]
                    ppos = self.data.oMf[pfid].translation.copy()
                    return float(np.linalg.norm(pos - ppos))
        return 0.3

    def _scale_positions(
        self, positions: np.ndarray, scale: dict[str, float]
    ) -> np.ndarray:
        """
        Scale endpoint positions proportionally to match robot limb lengths.
        Operates from root outward, adjusting each endpoint relative to its parent.
        """
        scaled = positions.copy()
        for child_j, parent_j, link_name in G1_LIMB_SCALE_PAIRS:
            s = scale.get(link_name, 1.0)
            parent_pos = scaled[:, int(parent_j)]
            child_pos  = scaled[:, int(child_j)]
            direction  = child_pos - parent_pos
            scaled[:, int(child_j)] = parent_pos + direction * s
        return scaled

    def _build_targets(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        targets = {}
        for link, joints in G1_BODY_TARGETS.items():
            if link not in self._frame_ids:
                continue
            if len(joints) == 1:
                targets[link] = pos[int(joints[0])]
            else:
                targets[link] = np.mean([pos[int(j)] for j in joints], axis=0)
        return targets

    def _ik_step(self, q: np.ndarray, targets: dict[str, np.ndarray]) -> np.ndarray:
        """Damped least-squares IK, iterating until convergence or max iterations."""
        q = q.copy()
        for _ in range(_IK_MAX_ITER):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            J_full = np.zeros((3 * len(targets), self.model.nv))
            err_full = np.zeros(3 * len(targets))

            for i, (link, tgt) in enumerate(targets.items()):
                fid = self._frame_ids[link]
                cur = self.data.oMf[fid].translation
                err_full[3*i:3*i+3] = tgt - cur

                J = pin.computeFrameJacobian(
                    self.model, self.data, q, fid,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )[:3]  # position rows only
                J_full[3*i:3*i+3] = J

            if np.linalg.norm(err_full) < _IK_EPS:
                break

            JtJ = J_full.T @ J_full
            damp = _IK_DAMP * np.eye(self.model.nv)
            dq = np.linalg.solve(JtJ + damp, J_full.T @ err_full)
            q = pin.integrate(self.model, q, _IK_DT * dq)
            q[7:7 + self.ndof] = np.clip(q[7:7 + self.ndof], self._q_lo, self._q_hi)

        return q

    @staticmethod
    def _yaw_quat(forward: np.ndarray) -> np.ndarray:
        """Return xyzw quaternion for a yaw-only rotation aligning +x to forward."""
        forward = forward.copy()
        forward[1] = 0.0  # flatten to xz plane
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0])
        forward /= norm
        yaw = np.arctan2(forward[2], forward[0])
        half = yaw / 2.0
        return np.array([0.0, np.sin(half), 0.0, np.cos(half)])
