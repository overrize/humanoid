"""
HMR2-based pose extractor: video → SMPL rotation matrices.
Bypasses IK entirely — each joint rotation comes directly from the model.
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import PoseExtractor as BasePoseExtractor
from ..nsf.format import NSFSequence, Joint

# SMPL 23-joint indices (body_pose, excludes global_orient pelvis)
_SMPL_IDX = {
    "l_hip": 0, "r_hip": 1,
    "spine1": 2,
    "l_knee": 3, "r_knee": 4,
    "spine2": 5,
    "l_ankle": 6, "r_ankle": 7,
    "spine3": 8,
    "l_foot": 9, "r_foot": 10,
    "neck": 11,
    "l_collar": 12, "r_collar": 13,
    "head": 14,
    "l_shoulder": 15, "r_shoulder": 16,
    "l_elbow": 17, "r_elbow": 18,
    "l_wrist": 19, "r_wrist": 20,
    "l_hand": 21, "r_hand": 22,
}


def _rotmat_to_nsf_joint(R33: np.ndarray) -> tuple[float, float, float]:
    """Extract (x, y, z) axis-angle from 3×3 rotation matrix."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R33).as_rotvec()


class SMPLExtractor(BasePoseExtractor):
    """
    Runs HMR2 on every frame of a video and stores:
      - smpl_rotmats : (T, 24, 3, 3)  — SMPL joint rotation matrices (0=global)
      - smpl_betas   : (T, 10)
      - smpl_trans   : (T, 3)
      - positions    : (T, 23, 3)     — NSF positions derived from FK
    Also populates an NSFSequence so the rest of the pipeline stays compatible.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._model_cfg = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from hmr2.models import load_hmr2
        except ImportError as e:
            raise ImportError("hmr2 not found. Run: pip install -e /home/rexcon/4D-Humans") from e

        # PyTorch ≥2.6 needs safe_globals for omegaconf types
        try:
            import omegaconf
            torch.serialization.add_safe_globals([
                omegaconf.DictConfig, omegaconf.ListConfig,
                omegaconf.dictconfig.DictConfig,
            ])
        except Exception:
            pass

        ckpt = self.checkpoint_path
        self._model, self._model_cfg = load_hmr2(ckpt) if ckpt else load_hmr2()
        self._model = self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    def extract(self, video_path: str, fps_out: float = 30.0) -> NSFSequence:
        """
        Run HMR2 frame-by-frame and return an NSFSequence.
        The NSFSequence positions come from SMPL FK; rotations are stored
        as quaternions on the sequence for use by SMPLRetargeter.
        """
        import cv2
        from hmr2.utils import recursive_to

        self._load_model()

        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps_out
        frame_skip = max(1, round(src_fps / fps_out))

        all_rotmats = []   # (T, 24, 3, 3)
        all_betas   = []   # (T, 10)
        all_trans   = []   # (T, 3)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            batch = self._preprocess_frame(frame)
            batch = recursive_to(batch, self.device)

            with torch.no_grad():
                out = self._model(batch)

            rotmat = out["pred_smpl_params"]["body_pose"].cpu().numpy()   # (1,23,3,3)
            global_orient = out["pred_smpl_params"]["global_orient"].cpu().numpy()  # (1,1,3,3)
            betas = out["pred_smpl_params"]["betas"].cpu().numpy()         # (1,10)
            # camera translation is in weak-perspective; use zeros as placeholder
            trans = np.zeros((1, 3))

            # Concatenate global_orient + body_pose → (1,24,3,3)
            full = np.concatenate([global_orient, rotmat], axis=1)
            all_rotmats.append(full[0])
            all_betas.append(betas[0])
            all_trans.append(trans[0])

            frame_idx += 1

        cap.release()

        if not all_rotmats:
            raise RuntimeError(f"No frames extracted from {video_path}")

        rotmats = np.stack(all_rotmats)   # (T,24,3,3)
        betas   = np.stack(all_betas)     # (T,10)
        trans   = np.stack(all_trans)     # (T,3)

        # Run FK to get joint positions
        positions = self._smpl_fk(rotmats, betas, trans)  # (T,23,3)

        # Convert rotmats to quaternions for NSFSequence storage
        from scipy.spatial.transform import Rotation
        T = len(rotmats)
        quats = Rotation.from_matrix(rotmats[:, 1:, :, :].reshape(-1, 3, 3)).as_quat()  # (T*23,4) xyzw
        quats = quats.reshape(T, 23, 4)
        # Convert xyzw → wxyz
        quats = np.concatenate([quats[:, :, 3:4], quats[:, :, :3]], axis=-1)

        seq = NSFSequence(
            fps=fps_out,
            positions=positions,
            rotations=quats,
            contacts=np.zeros((T, 4), dtype=bool),
        )
        # Stash raw SMPL params on the sequence for the retargeter
        seq._smpl_rotmats = rotmats   # (T,24,3,3)  0=global_orient, 1-23=body_pose
        seq._smpl_betas   = betas
        seq._smpl_trans   = trans
        return seq

    # ------------------------------------------------------------------
    def _preprocess_frame(self, bgr_frame: np.ndarray) -> dict:
        """Crop + normalise a single BGR frame into the hmr2 input dict."""
        import cv2
        h, w = bgr_frame.shape[:2]
        # Simple centre crop to square, then resize to 256×256
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = bgr_frame[y0:y0+side, x0:x0+side]
        crop = cv2.resize(crop, (256, 256))
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        rgb  = (rgb - mean) / std
        img_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
        # HMR2 expects bbox in XYXY format — use full image
        bbox = torch.tensor([[0, 0, 256, 256]], dtype=torch.float32)
        return {"img": img_t, "bbox": bbox, "img_size": torch.tensor([[256, 256]])}

    def _smpl_fk(self, rotmats: np.ndarray, betas: np.ndarray, trans: np.ndarray) -> np.ndarray:
        """
        Run SMPL FK to get joint positions (T,23,3).
        Falls back to zero positions if SMPL neutral model is unavailable.
        """
        try:
            import smplx
            from hmr2.models import check_smpl_exists
            check_smpl_exists()
            body = smplx.create(
                model_type="smpl",
                gender="neutral",
                num_betas=10,
                batch_size=len(rotmats),
            )
            from scipy.spatial.transform import Rotation
            T = len(rotmats)
            body_pose_aa = Rotation.from_matrix(rotmats[:, 1:, :, :].reshape(-1, 3, 3)).as_rotvec()
            body_pose_aa = body_pose_aa.reshape(T, 69)
            global_aa = Rotation.from_matrix(rotmats[:, 0, :, :]).as_rotvec()
            out = body(
                global_orient=torch.from_numpy(global_aa).float(),
                body_pose=torch.from_numpy(body_pose_aa).float(),
                betas=torch.from_numpy(betas).float(),
                transl=torch.from_numpy(trans).float(),
                return_verts=False,
            )
            joints = out.joints.detach().numpy()[:, :23, :]   # (T,23,3)
            # Ground align
            min_y = joints[:, :, 1].min()
            joints[:, :, 1] -= min_y
            return joints
        except Exception as e:
            warnings.warn(f"SMPL FK failed ({e}); using zero positions. "
                          "Download SMPL_NEUTRAL.pkl from https://smpl.is.tue.mpg.de/")
            T = len(rotmats)
            return np.zeros((T, 23, 3), dtype=np.float32)
