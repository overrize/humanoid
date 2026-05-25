"""Abstract base for all pose extractors."""

from abc import ABC, abstractmethod
from pathlib import Path
from ..nsf.format import NSFSequence


class PoseExtractor(ABC):
    """Convert a video or mocap file to an NSFSequence."""

    @abstractmethod
    def extract(self, source: str | Path) -> NSFSequence:
        """
        Args:
            source: Path to video file, BVH file, or similar.
        Returns:
            NSFSequence in world-frame coordinates.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, e.g. 'mediapipe', 'motionbert', 'bvh'."""
        ...
