"""
BVH / mocap file extractor.
Reads a BVH file and maps its skeleton to NSF format.
Supports BVH files from CMU Mocap, LAFAN1, and similar datasets.
"""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base import PoseExtractor
from ..nsf.format import NSFSequence, Joint, NUM_JOINTS, PARENT


# BVH joint name patterns → NSF joint
# Uses substring matching so it works across naming conventions.
_BVH_NAME_MAP: list[tuple[str, Joint]] = [
    ("hip",             Joint.ROOT),
    ("spine",           Joint.SPINE),
    ("spine1",          Joint.CHEST),
    ("neck",            Joint.NECK),
    ("head",            Joint.HEAD),
    ("leftshoulder",    Joint.L_SHOULDER),
    ("rightshoulder",   Joint.R_SHOULDER),
    ("leftarm",         Joint.L_SHOULDER),
    ("rightarm",        Joint.R_SHOULDER),
    ("leftforearm",     Joint.L_ELBOW),
    ("rightforearm",    Joint.R_ELBOW),
    ("lefthand",        Joint.L_WRIST),
    ("righthand",       Joint.R_WRIST),
    ("leftupleg",       Joint.L_HIP),
    ("rightupleg",      Joint.R_HIP),
    ("leftleg",         Joint.L_KNEE),
    ("rightleg",        Joint.R_KNEE),
    ("leftfoot",        Joint.L_ANKLE),
    ("rightfoot",       Joint.R_ANKLE),
    ("lefttoebase",     Joint.L_TOE),
    ("righttoebase",    Joint.R_TOE),
]


def _match_joint(bvh_name: str) -> Joint | None:
    key = bvh_name.lower().replace("_", "").replace(" ", "")
    for pattern, nsf_joint in _BVH_NAME_MAP:
        if pattern in key:
            return nsf_joint
    return None


class BVHExtractor(PoseExtractor):

    @property
    def name(self) -> str:
        return "bvh"

    def extract(self, source: str | Path) -> NSFSequence:
        try:
            import bvhio  # pip install bvhio
        except ImportError:
            raise ImportError("pip install bvhio")

        source = Path(source)
        reader = bvhio.readAsHierarchy(str(source))
        fps = reader.FrameTime and (1.0 / reader.FrameTime) or 30.0
        T = reader.FrameCount

        joint_names = [j.Name for j in reader.filter("*")]
        positions_raw = {}
        for jnt in reader.filter("*"):
            frames = []
            for f in range(T):
                reader.readPose(f)
                pos = np.array(jnt.PositionWorld, dtype=np.float32)
                frames.append(pos)
            positions_raw[jnt.Name] = np.stack(frames)  # (T, 3)

        positions = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
        covered = set()
        for bvh_name, pts in positions_raw.items():
            nsf_joint = _match_joint(bvh_name)
            if nsf_joint is not None and nsf_joint not in covered:
                positions[:, int(nsf_joint)] = pts
                covered.add(nsf_joint)

        # Fill any missing joints by interpolating from parent
        for jnt in Joint:
            if jnt not in covered and PARENT[jnt] is not None:
                positions[:, int(jnt)] = positions[:, int(PARENT[jnt])]

        # Ground align
        foot_min = np.minimum(
            positions[:, Joint.L_FOOT, 1],
            positions[:, Joint.R_FOOT, 1],
        ).min()
        positions[:, :, 1] -= foot_min

        # Scale to meters if values look like cm
        if positions[:, Joint.HEAD, 1].mean() > 10.0:
            positions /= 100.0

        rotations = np.zeros((T, NUM_JOINTS, 4), dtype=np.float32)
        rotations[:, :, 0] = 1.0

        contacts = self._detect_contacts(positions, fps)

        return NSFSequence(
            positions=positions,
            rotations=rotations,
            contacts=contacts,
            fps=fps,
            source=self.name,
            name=source.stem,
        )

    def _detect_contacts(self, positions: np.ndarray, fps: float) -> np.ndarray:
        T = positions.shape[0]
        contacts = np.zeros((T, 4), dtype=np.float32)
        dt = 1.0 / fps
        foot_joints = [Joint.L_FOOT, Joint.R_FOOT, Joint.L_TOE, Joint.R_TOE]
        for col, jnt in enumerate(foot_joints):
            pos = positions[:, int(jnt)]
            vel = np.zeros(T)
            vel[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=-1) / dt
            height = pos[:, 1]
            contacts[:, col] = ((vel < 0.15) & (height < 0.05)).astype(np.float32)
        return contacts
