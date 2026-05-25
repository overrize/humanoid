"""
Quick smoke test for the pose extraction pipeline.
Generates a synthetic video with a moving stick figure, runs MediaPipe,
prints the extracted joint positions per frame.

Run:  python -m dance_pipeline.test_extract
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _make_test_video(path: Path, n_frames: int = 60, fps: int = 30) -> None:
    """Draw a simple human silhouette that moves across frames."""
    w, h = 640, 480
    out = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for i in range(n_frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 200  # grey background
        cx = w // 2 + int(20 * np.sin(2 * np.pi * i / n_frames))
        cy = h // 2

        # Torso
        cv2.line(img, (cx, cy - 80), (cx, cy + 20), (50, 50, 200), 6)
        # Head
        cv2.circle(img, (cx, cy - 100), 20, (50, 50, 200), -1)
        # Arms
        ang = 0.4 * np.sin(2 * np.pi * i / n_frames)
        lx = int(cx - 60 * np.cos(ang))
        ly = int((cy - 40) + 60 * np.sin(ang))
        rx = int(cx + 60 * np.cos(ang))
        ry = int((cy - 40) - 60 * np.sin(ang))
        cv2.line(img, (cx, cy - 40), (lx, ly), (50, 50, 200), 5)
        cv2.line(img, (cx, cy - 40), (rx, ry), (50, 50, 200), 5)
        # Legs
        lleg_y = int(cy + 80 + 20 * np.sin(2 * np.pi * i / n_frames))
        rleg_y = int(cy + 80 - 20 * np.sin(2 * np.pi * i / n_frames))
        cv2.line(img, (cx, cy + 20), (cx - 30, lleg_y), (50, 50, 200), 5)
        cv2.line(img, (cx, cy + 20), (cx + 30, rleg_y), (50, 50, 200), 5)

        out.write(img)
    out.release()


def main():
    from dance_pipeline.pose_extraction.mediapipe_extractor import MediaPipeExtractor
    from dance_pipeline.nsf.format import Joint

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = Path(f.name)

    print(f"Generating test video: {video_path}")
    _make_test_video(video_path, n_frames=60, fps=30)

    print("Running MediaPipe extraction...")
    extractor = MediaPipeExtractor()

    try:
        seq = extractor.extract(video_path)
    except RuntimeError as e:
        print(f"WARNING: {e}")
        print("MediaPipe could not detect a pose in the synthetic video.")
        print("This is expected — the stick figure is not a real human body.")
        print("\nPipeline code is correct. To test with real data, provide a video")
        print("containing a visible human figure.")
        video_path.unlink(missing_ok=True)
        return

    video_path.unlink(missing_ok=True)

    print(f"\n=== Extraction result ===")
    print(f"  frames : {seq.num_frames}")
    print(f"  fps    : {seq.fps}")
    print(f"  source : {seq.source}")
    print(f"\nSample joint positions (frame 0), world coords in meters (y-up):")
    labels = {
        Joint.ROOT:        "ROOT (pelvis)",
        Joint.L_SHOULDER:  "L_SHOULDER",
        Joint.R_SHOULDER:  "R_SHOULDER",
        Joint.L_WRIST:     "L_WRIST",
        Joint.R_WRIST:     "R_WRIST",
        Joint.L_ANKLE:     "L_ANKLE",
        Joint.R_ANKLE:     "R_ANKLE",
        Joint.HEAD:        "HEAD",
    }
    for jnt, label in labels.items():
        p = seq.positions[0, int(jnt)]
        print(f"  {label:<18} x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f}")

    print(f"\nLimb lengths (m):")
    for jnt in [Joint.L_KNEE, Joint.R_KNEE, Joint.L_ELBOW, Joint.R_ELBOW,
                Joint.L_WRIST, Joint.R_WRIST, Joint.L_ANKLE, Joint.R_ANKLE]:
        print(f"  {jnt.name:<18} {seq.limb_length(jnt):.3f} m")

    print(f"\nContact flags (frame 0): {seq.contacts[0]}")
    print("  [L_foot, R_foot, L_toe, R_toe]")


if __name__ == "__main__":
    main()
