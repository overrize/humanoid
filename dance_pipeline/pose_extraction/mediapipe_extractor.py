"""
MediaPipe Pose extractor (Tasks API, mediapipe >= 0.10).
Maps MediaPipe's 33-landmark world skeleton to NSF's 23-joint format.
"""

from pathlib import Path
import numpy as np

from .base import PoseExtractor
from ..nsf.format import NSFSequence, Joint, NUM_JOINTS

# Default model path relative to this file
_DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "pose_landmarker_lite.task"

# MediaPipe PoseLandmark indices (Tasks API)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
class _MP:
    NOSE           = 0
    L_SHOULDER     = 11
    R_SHOULDER     = 12
    L_ELBOW        = 13
    R_ELBOW        = 14
    L_WRIST        = 15
    R_WRIST        = 16
    L_PINKY        = 17
    R_PINKY        = 18
    L_INDEX        = 19
    R_INDEX        = 20
    L_HIP          = 23
    R_HIP          = 24
    L_KNEE         = 25
    R_KNEE         = 26
    L_ANKLE        = 27
    R_ANKLE        = 28
    L_HEEL         = 29
    R_HEEL         = 30
    L_FOOT_INDEX   = 31
    R_FOOT_INDEX   = 32

_CONTACT_VEL_THRESH    = 0.15   # m/s
_CONTACT_HEIGHT_THRESH = 0.05   # m


class MediaPipeExtractor(PoseExtractor):

    def __init__(
        self,
        model_path: str | Path | None = None,
        target_fps: float = 30.0,
    ):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL
        self._target_fps = target_fps

    @property
    def name(self) -> str:
        return "mediapipe"

    def extract(self, source: str | Path) -> NSFSequence:
        try:
            import mediapipe as mp
            import cv2
        except ImportError:
            raise ImportError("pip install mediapipe opencv-python")

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self._model_path}\n"
                "Download pose_landmarker_lite.task from mediapipe model page."
            )

        source = Path(source)
        cap = cv2.VideoCapture(str(source))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self._target_fps

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        raw_world: list[np.ndarray] = []   # (33, 3) world landmarks per frame

        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_idx * 1000 / src_fps)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frame_idx += 1

                if result.pose_world_landmarks:
                    lm = result.pose_world_landmarks[0]  # first person
                    pts = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
                    raw_world.append(pts)
                else:
                    # Repeat last known pose to avoid gaps
                    if raw_world:
                        raw_world.append(raw_world[-1].copy())

        cap.release()

        if not raw_world:
            raise RuntimeError(f"No pose detected in {source}")

        landmarks = np.stack(raw_world)   # (T, 33, 3)
        positions, rotations = self._build_nsf_skeleton(landmarks)
        contacts = self._detect_contacts(positions, src_fps)

        return NSFSequence(
            positions=positions,
            rotations=rotations,
            contacts=contacts,
            fps=src_fps,
            source=self.name,
            name=source.stem,
        )

    def _build_nsf_skeleton(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = landmarks.shape[0]
        pos = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)

        # Direct mappings
        pos[:, Joint.L_HIP]      = landmarks[:, _MP.L_HIP]
        pos[:, Joint.R_HIP]      = landmarks[:, _MP.R_HIP]
        pos[:, Joint.L_KNEE]     = landmarks[:, _MP.L_KNEE]
        pos[:, Joint.R_KNEE]     = landmarks[:, _MP.R_KNEE]
        pos[:, Joint.L_ANKLE]    = landmarks[:, _MP.L_ANKLE]
        pos[:, Joint.R_ANKLE]    = landmarks[:, _MP.R_ANKLE]
        pos[:, Joint.L_SHOULDER] = landmarks[:, _MP.L_SHOULDER]
        pos[:, Joint.R_SHOULDER] = landmarks[:, _MP.R_SHOULDER]
        pos[:, Joint.L_ELBOW]    = landmarks[:, _MP.L_ELBOW]
        pos[:, Joint.R_ELBOW]    = landmarks[:, _MP.R_ELBOW]
        pos[:, Joint.L_WRIST]    = landmarks[:, _MP.L_WRIST]
        pos[:, Joint.R_WRIST]    = landmarks[:, _MP.R_WRIST]
        pos[:, Joint.L_HAND]     = (landmarks[:, _MP.L_PINKY] + landmarks[:, _MP.L_INDEX]) / 2
        pos[:, Joint.R_HAND]     = (landmarks[:, _MP.R_PINKY] + landmarks[:, _MP.R_INDEX]) / 2
        pos[:, Joint.L_FOOT]     = landmarks[:, _MP.L_HEEL]
        pos[:, Joint.R_FOOT]     = landmarks[:, _MP.R_HEEL]
        pos[:, Joint.L_TOE]      = landmarks[:, _MP.L_FOOT_INDEX]
        pos[:, Joint.R_TOE]      = landmarks[:, _MP.R_FOOT_INDEX]
        pos[:, Joint.HEAD]       = landmarks[:, _MP.NOSE]

        # Derived joints
        l_hip = landmarks[:, _MP.L_HIP]
        r_hip = landmarks[:, _MP.R_HIP]
        l_sh  = landmarks[:, _MP.L_SHOULDER]
        r_sh  = landmarks[:, _MP.R_SHOULDER]

        pos[:, Joint.ROOT]  = (l_hip + r_hip) / 2
        pos[:, Joint.CHEST] = (l_sh + r_sh) / 2
        pos[:, Joint.SPINE] = (pos[:, Joint.ROOT] + pos[:, Joint.CHEST]) / 2
        pos[:, Joint.NECK]  = pos[:, Joint.CHEST] + np.array([[0.0, 0.05, 0.0]])

        # MediaPipe world landmarks: y=down, z=forward → convert to y-up
        # Swap y and z, negate new y (was z)
        pos = pos[:, :, [0, 2, 1]]   # x,z,y
        pos[:, :, 1] *= -1            # flip new y (was z, now up)

        # Ground align: shift so min foot height = 0
        foot_min = np.minimum(
            pos[:, Joint.L_FOOT, 1],
            pos[:, Joint.R_FOOT, 1],
        ).min()
        pos[:, :, 1] -= foot_min

        # Identity quaternions
        rot = np.zeros((T, NUM_JOINTS, 4), dtype=np.float32)
        rot[:, :, 0] = 1.0  # w=1, wxyz

        return pos, rot

    def _detect_contacts(self, positions: np.ndarray, fps: float) -> np.ndarray:
        T = positions.shape[0]
        contacts = np.zeros((T, 4), dtype=np.float32)
        dt = 1.0 / fps
        foot_joints = [Joint.L_FOOT, Joint.R_FOOT, Joint.L_TOE, Joint.R_TOE]
        for col, jnt in enumerate(foot_joints):
            p = positions[:, int(jnt)]
            vel = np.zeros(T)
            vel[1:] = np.linalg.norm(np.diff(p, axis=0), axis=-1) / dt
            contacts[:, col] = (
                (vel < _CONTACT_VEL_THRESH) &
                (p[:, 1] < _CONTACT_HEIGHT_THRESH)
            ).astype(np.float32)
        return contacts
