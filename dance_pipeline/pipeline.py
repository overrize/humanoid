"""
End-to-end dance import pipeline.

Usage:
    python -m dance_pipeline.pipeline \\
        --input  my_dance.mp4 \\
        --robot  g1 \\
        --urdf   /path/to/g1_29dof_rev_1_0.urdf \\
        --mesh   /path/to/g1_description \\
        --output motions/my_dance_g1.npz
"""

import argparse
from pathlib import Path

from .pose_extraction.mediapipe_extractor import MediaPipeExtractor
from .pose_extraction.bvh_extractor import BVHExtractor
from .pose_extraction.smpl_extractor import SMPLExtractor
from .nsf.io import save_nsf, load_nsf
from .retargeting.urdf_retargeter import URDFRetargeter
from .retargeting.smpl_retargeter import SMPLRetargeter
from .retargeting.geo_retargeter import GeoRetargeter
from .motion_builder import build_npz

_EXTRACTORS = {
    "mediapipe": MediaPipeExtractor,
    "bvh":       BVHExtractor,
    "smpl":      SMPLExtractor,
}

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_BVH_EXTS   = {".bvh"}
_NSF_EXTS   = {".npz"}


def auto_extractor(source: Path, extractor: str = "auto"):
    ext = source.suffix.lower()
    if extractor == "smpl":
        return SMPLExtractor()
    if ext in _VIDEO_EXTS:
        return MediaPipeExtractor()
    if ext in _BVH_EXTS:
        return BVHExtractor()
    raise ValueError(f"Cannot determine extractor for extension {ext!r}")


def run(
    input_path:  str | Path,
    urdf_path:   str | Path,
    mesh_dir:    str | Path,
    output_path: str | Path,
    robot_name:  str = "g1",
    nsf_cache:   str | Path | None = None,
    extractor:   str = "auto",
    skip_extract: bool = False,
) -> Path:
    """
    Full pipeline: input → NSF → robot joint angles → NPZ.

    Args:
        input_path:   Video, BVH, or pre-computed NSF file.
        urdf_path:    Robot URDF path.
        mesh_dir:     Mesh directory for URDF.
        output_path:  Where to write the output NPZ.
        robot_name:   Label for the robot (used in retargeter).
        nsf_cache:    If set, save/load NSF at this path (skip extraction if exists).
        extractor:    "auto", "mediapipe", or "bvh".
        skip_extract: If True and nsf_cache exists, skip extraction step.
    Returns:
        Path to the written NPZ.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    # ── Step 1: Pose extraction → NSF ───────────────────────────────────────
    nsf_path = Path(nsf_cache) if nsf_cache else None
    if nsf_path and nsf_path.exists() and skip_extract:
        print(f"[pipeline] Loading cached NSF from {nsf_path}")
        seq = load_nsf(nsf_path)
    elif input_path.suffix.lower() in _NSF_EXTS:
        print(f"[pipeline] Loading NSF directly from {input_path}")
        seq = load_nsf(input_path)
    else:
        ext_cls = _EXTRACTORS.get(extractor) if extractor not in ("auto",) else None
        extr = ext_cls() if ext_cls else auto_extractor(input_path, extractor)
        print(f"[pipeline] Extracting pose with {extr.name} from {input_path}")
        seq = extr.extract(input_path)
        if nsf_path:
            save_nsf(seq, nsf_path)
            print(f"[pipeline] NSF saved to {nsf_path}")

    print(f"[pipeline] NSF: {seq.num_frames} frames @ {seq.fps} fps  ({seq.duration:.1f}s)  source={seq.source}")

    # ── Step 2: Retargeting → robot joint angles ─────────────────────────────
    print(f"[pipeline] Retargeting to {robot_name} …")
    if extractor == "smpl" or hasattr(seq, "_smpl_rotmats"):
        retargeter = SMPLRetargeter()
    else:
        retargeter = GeoRetargeter()
    result = retargeter.retarget(seq)
    print(f"[pipeline] Retargeted: {result['dof_positions'].shape[0]} frames, "
          f"{result['dof_positions'].shape[1]} DOFs")

    # ── Step 3: FK + NPZ ─────────────────────────────────────────────────────
    print(f"[pipeline] Building NPZ → {output_path}")
    out = build_npz(result, urdf_path, mesh_dir, output_path)
    print(f"[pipeline] Done: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Dance import pipeline")
    parser.add_argument("--input",   required=True, help="Video / BVH / NSF file")
    parser.add_argument("--urdf",    required=True, help="Robot URDF path")
    parser.add_argument("--mesh",    required=True, help="Mesh directory")
    parser.add_argument("--output",  required=True, help="Output NPZ path")
    parser.add_argument("--robot",   default="g1",  help="Robot name label")
    parser.add_argument("--nsf-cache", default=None, help="Save/load NSF at this path")
    parser.add_argument("--extractor", default="auto",
                        choices=["auto", "mediapipe", "bvh", "smpl"])
    parser.add_argument("--skip-extract", action="store_true",
                        help="Use cached NSF if available")
    args = parser.parse_args()

    run(
        input_path   = args.input,
        urdf_path    = args.urdf,
        mesh_dir     = args.mesh,
        output_path  = args.output,
        robot_name   = args.robot,
        nsf_cache    = args.nsf_cache,
        extractor    = args.extractor,
        skip_extract = args.skip_extract,
    )


if __name__ == "__main__":
    main()
