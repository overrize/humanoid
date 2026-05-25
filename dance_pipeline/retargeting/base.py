"""Abstract retargeter interface."""

from abc import ABC, abstractmethod
import numpy as np
from ..nsf.format import NSFSequence


class Retargeter(ABC):
    """Map an NSFSequence to robot joint angles + root pose."""

    @abstractmethod
    def retarget(self, seq: NSFSequence) -> dict:
        """
        Args:
            seq: Human motion in NSF format.
        Returns:
            dict with keys:
                root_positions  (T, 3)    — pelvis position, world frame
                root_rotations  (T, 4)    — pelvis orientation, wxyz quaternion
                dof_positions   (T, N)    — joint angles in robot DOF order
                dof_velocities  (T, N)    — joint angular velocities
                dof_names       list[str] — DOF name for each column
                fps             float
        """
        ...

    @property
    @abstractmethod
    def robot_name(self) -> str:
        ...
