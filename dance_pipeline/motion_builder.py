"""
Build a MotionLoader-compatible NPZ from retargeter output.
Runs Pinocchio FK to fill body_positions/body_rotations,
then computes body velocities — mirrors what data_convert.py did,
but as a reusable library function.
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

import pinocchio as pin


def build_npz(
    retarget_result: dict,
    urdf_path:  str | Path,
    mesh_dir:   str | Path,
    output_path: str | Path,
    body_links: list[str] | None = None,
) -> Path:
    """
    Args:
        retarget_result: Output dict from URDFRetargeter.retarget().
        urdf_path:       Robot URDF.
        mesh_dir:        Mesh directory for URDF.
        output_path:     Where to write the .npz file.
        body_links:      Which robot links to record in body_positions/rotations.
                         Defaults to the standard AMP body set for G1.
    Returns:
        Path to the written NPZ file.
    """
    if body_links is None:
        body_links = [
            "pelvis",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_elbow_link",
            "right_elbow_link",
            "right_hip_yaw_link",
            "left_hip_yaw_link",
            "right_rubber_hand",
            "left_rubber_hand",
            "right_ankle_roll_link",
            "left_ankle_roll_link",
        ]

    root_positions  = retarget_result["root_positions"]   # (T, 3)
    root_rotations  = retarget_result["root_rotations"]   # (T, 4) wxyz
    dof_positions   = retarget_result["dof_positions"]    # (T, N)
    dof_velocities  = retarget_result["dof_velocities"]   # (T, N)
    dof_names       = retarget_result["dof_names"]
    fps             = float(retarget_result["fps"])
    dt              = 1.0 / fps
    T, N            = dof_positions.shape

    robot = pin.RobotWrapper.BuildFromURDF(
        str(urdf_path), str(mesh_dir), pin.JointModelFreeFlyer()
    )
    model = robot.model
    data  = robot.data

    frame_ids = [model.getFrameId(name) for name in body_links]
    B = len(body_links)

    body_positions         = np.zeros((T, B, 3), dtype=np.float32)
    body_rotations         = np.zeros((T, B, 4), dtype=np.float32)
    body_linear_velocities = np.zeros((T, B, 3), dtype=np.float32)
    body_angular_velocities= np.zeros((T, B, 3), dtype=np.float32)

    q = pin.neutral(model)

    for t in range(T):
        # Free-flyer: position
        q[0:3] = root_positions[t]
        # root_rotations is wxyz → pinocchio wants xyzw
        wx, wy, wz, ww = root_rotations[t, 1], root_rotations[t, 2], root_rotations[t, 3], root_rotations[t, 0]
        q[3:7] = [wx, wy, wz, ww]
        # Joint angles
        q[7:7 + N] = dof_positions[t]

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for j, fid in enumerate(frame_ids):
            tf = data.oMf[fid]
            body_positions[t, j] = tf.translation
            quat = pin.Quaternion(tf.rotation)
            body_rotations[t, j] = [quat.w, quat.x, quat.y, quat.z]  # wxyz

    # Body linear velocities: central differences
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0]    = (body_positions[1]  - body_positions[0])   / dt
    body_linear_velocities[-1]   = (body_positions[-1] - body_positions[-2])  / dt
    body_linear_velocities = gaussian_filter1d(body_linear_velocities, sigma=1, axis=0)

    # Body angular velocities
    for j in range(B):
        quats = body_rotations[:, j, :]  # wxyz
        av = np.zeros((T, 3), dtype=np.float32)
        for t in range(1, T - 1):
            av[t] = _angular_velocity(quats[t - 1], quats[t + 1], 2 * dt)
        av[0]  = _angular_velocity(quats[0],  quats[1],  dt)
        av[-1] = _angular_velocity(quats[-2], quats[-1], dt)
        body_angular_velocities[:, j] = gaussian_filter1d(av, sigma=1, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        fps=fps,
        dof_names=np.array(dof_names, dtype=np.str_),
        body_names=np.array(body_links, dtype=np.str_),
        dof_positions=dof_positions.astype(np.float32),
        dof_velocities=dof_velocities.astype(np.float32),
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    return output_path


def _angular_velocity(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity from two wxyz quaternions."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    # q_rel = inv(q0) * q1  (conjugate since unit quat)
    qr_w =  w0*w1 + x0*x1 + y0*y1 + z0*z1
    qr_x = -x0*w1 + w0*x1 - z0*y1 + y0*z1
    qr_y = -y0*w1 + z0*x1 + w0*y1 - x0*z1
    qr_z = -z0*w1 - y0*x1 + x0*y1 + w0*z1
    qr_w = np.clip(qr_w, -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(qr_w))
    sin_h = np.sqrt(max(1.0 - qr_w**2, 1e-12))
    if sin_h < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([qr_x, qr_y, qr_z]) / sin_h
    if qr_w < 0:
        angle = -angle
    return (angle / dt * axis).astype(np.float32)
