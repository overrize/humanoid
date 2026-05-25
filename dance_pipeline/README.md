# Dance Pipeline

Video/mocap → robot motion NPZ pipeline for humanoid dance imitation.

## Architecture

```
Video / BVH / SMPL
    ↓ pose_extraction/       — MediaPipe or BVH extractor → NSF
NSF (Normalized Skeleton Format, robot-agnostic)
    ↓ retargeting/           — URDF-aware retargeter → joint angles
Robot joint angle sequence
    ↓ motion_builder.py      — Pinocchio FK → NPZ
NPZ  →  AMP training / RViz preview
```

## Quick Start

```bash
# Full pipeline: video → NPZ
python -m dance_pipeline.pipeline \
    --input  dance.mp4 \
    --urdf   /path/to/robot.urdf \
    --mesh   /path/to/meshes \
    --output motions/dance.npz

# Visualize NSF skeleton (browser)
python -m dance_pipeline.visualize_nsf --file dance.nsf.npz --backend plotly

# Preview on robot in RViz
bash dance_pipeline/launch_rviz.sh motions/dance.npz

# Validate retargeting quality
python -m dance_pipeline.validate_retargeting
```

## Modules

| Module | Description |
|--------|-------------|
| `nsf/` | Normalized Skeleton Format — 23-joint robot-agnostic representation |
| `pose_extraction/` | MediaPipe (video) and BVH (mocap) extractors |
| `retargeting/` | URDF-aware retargeter using Pinocchio IK |
| `motion_builder.py` | FK + NPZ writer, compatible with existing MotionLoader |
| `visualize_nsf.py` | Matplotlib / Plotly skeleton visualizer |
| `rviz_publisher.py` | ROS joint_state publisher for RViz preview |
| `validate_retargeting.py` | Round-trip quality validation |

## Roadmap

- [ ] SMPL-based retargeting (replace IK with rotation mapping)
- [ ] motion_library: metadata, multi-robot compatibility tags
- [ ] Community platform: upload / preview / share dance sequences
