"""
Normalized Skeleton Format (NSF)
Robot-agnostic intermediate representation for human motion.

Joint layout follows a subset of SMPL-X / MediaPipe Pose conventions,
reduced to the 23 joints most relevant for whole-body dance retargeting.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np


class Joint(IntEnum):
    """Canonical joint indices in NSF."""
    ROOT        = 0   # pelvis center
    L_HIP       = 1
    R_HIP       = 2
    SPINE       = 3
    L_KNEE      = 4
    R_KNEE      = 5
    CHEST       = 6
    L_ANKLE     = 7
    R_ANKLE     = 8
    NECK        = 9
    L_SHOULDER  = 10
    R_SHOULDER  = 11
    HEAD        = 12
    L_ELBOW     = 13
    R_ELBOW     = 14
    L_WRIST     = 15
    R_WRIST     = 16
    L_FOOT      = 17
    R_FOOT      = 18
    L_HAND      = 19
    R_HAND      = 20
    L_TOE       = 21
    R_TOE       = 22

NUM_JOINTS = len(Joint)

# Parent joint for each joint (for limb-length computation)
PARENT = {
    Joint.ROOT:       None,
    Joint.L_HIP:      Joint.ROOT,
    Joint.R_HIP:      Joint.ROOT,
    Joint.SPINE:      Joint.ROOT,
    Joint.L_KNEE:     Joint.L_HIP,
    Joint.R_KNEE:     Joint.R_HIP,
    Joint.CHEST:      Joint.SPINE,
    Joint.L_ANKLE:    Joint.L_KNEE,
    Joint.R_ANKLE:    Joint.R_KNEE,
    Joint.NECK:       Joint.CHEST,
    Joint.L_SHOULDER: Joint.CHEST,
    Joint.R_SHOULDER: Joint.CHEST,
    Joint.HEAD:       Joint.NECK,
    Joint.L_ELBOW:    Joint.L_SHOULDER,
    Joint.R_ELBOW:    Joint.R_SHOULDER,
    Joint.L_WRIST:    Joint.L_ELBOW,
    Joint.R_WRIST:    Joint.R_ELBOW,
    Joint.L_FOOT:     Joint.L_ANKLE,
    Joint.R_FOOT:     Joint.R_ANKLE,
    Joint.L_HAND:     Joint.L_WRIST,
    Joint.R_HAND:     Joint.R_WRIST,
    Joint.L_TOE:      Joint.L_FOOT,
    Joint.R_TOE:      Joint.R_FOOT,
}

# Foot contact joints used for ground constraint
FOOT_JOINTS = [Joint.L_FOOT, Joint.R_FOOT, Joint.L_TOE, Joint.R_TOE]


@dataclass
class NSFSequence:
    """
    A single motion sequence in Normalized Skeleton Format.

    Attributes:
        positions:  (T, NUM_JOINTS, 3)  — joint positions in world frame, meters
        rotations:  (T, NUM_JOINTS, 4)  — joint orientations as wxyz quaternion
        contacts:   (T, 4)              — foot contact flags [l_foot, r_foot, l_toe, r_toe]
        fps:        float               — frames per second
        source:     str                 — origin tag, e.g. "mediapipe", "bvh", "smpl"
        name:       str                 — motion clip name
    """
    positions:  np.ndarray
    rotations:  np.ndarray
    contacts:   np.ndarray
    fps:        float
    source:     str = "unknown"
    name:       str = ""

    def __post_init__(self):
        T = self.positions.shape[0]
        assert self.positions.shape  == (T, NUM_JOINTS, 3), \
            f"positions shape mismatch: {self.positions.shape}"
        assert self.rotations.shape  == (T, NUM_JOINTS, 4), \
            f"rotations shape mismatch: {self.rotations.shape}"
        assert self.contacts.shape   == (T, 4), \
            f"contacts shape mismatch: {self.contacts.shape}"

    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    def limb_length(self, child: Joint) -> float:
        """Average length of the bone connecting child to its parent."""
        parent = PARENT[child]
        if parent is None:
            return 0.0
        diff = self.positions[:, int(child)] - self.positions[:, int(parent)]
        return float(np.linalg.norm(diff, axis=-1).mean())

    def limb_lengths(self) -> dict[Joint, float]:
        return {j: self.limb_length(j) for j in Joint if PARENT[j] is not None}
