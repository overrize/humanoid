"""
Retargeting quality validation.

Pipeline:
  G1_dance.npz (body_positions) → NSF (inverse map) → URDFRetargeter → dof_positions
                                                                              ↓
                                                            compare vs original dof_positions

Run:
    python -m dance_pipeline.validate_retargeting
"""

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dance_pipeline.nsf.format import NSFSequence, Joint, NUM_JOINTS
from dance_pipeline.retargeting.urdf_retargeter import URDFRetargeter

URDF_PATH = Path("/home/rexcon/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf")
MESH_DIR  = Path("/home/rexcon/unitree_ros/robots/g1_description")
NPZ_PATH  = Path("/home/rexcon/humanoid_amp/motions/G1_dance.npz")


# G1 body_name → NSF Joint (from joint_map, only what's in body_names)
BODY_TO_NSF = {
    "pelvis":                   Joint.ROOT,
    "left_hip_yaw_link":        Joint.L_HIP,
    "right_hip_yaw_link":       Joint.R_HIP,
    "left_knee_link":           Joint.L_KNEE,
    "right_knee_link":          Joint.R_KNEE,
    "left_ankle_roll_link":     Joint.L_ANKLE,
    "right_ankle_roll_link":    Joint.R_ANKLE,
    "torso_link":               Joint.CHEST,
    "left_shoulder_pitch_link": Joint.L_SHOULDER,
    "right_shoulder_pitch_link":Joint.R_SHOULDER,
    "left_elbow_link":          Joint.L_ELBOW,
    "right_elbow_link":         Joint.R_ELBOW,
    "left_rubber_hand":         Joint.L_WRIST,
    "right_rubber_hand":        Joint.R_WRIST,
}


def body_positions_to_nsf(body_positions: np.ndarray, body_names: list[str], fps: float) -> NSFSequence:
    """Build NSFSequence from G1 body_positions array."""
    T = body_positions.shape[0]
    positions = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)

    name_to_idx = {n: i for i, n in enumerate(body_names)}

    for body_name, nsf_joint in BODY_TO_NSF.items():
        if body_name in name_to_idx:
            positions[:, int(nsf_joint)] = body_positions[:, name_to_idx[body_name]]

    # Fill derived joints from available data
    positions[:, Joint.SPINE] = (positions[:, Joint.ROOT] + positions[:, Joint.CHEST]) / 2
    positions[:, Joint.NECK]  = positions[:, Joint.CHEST] + np.array([0, 0.1, 0])
    positions[:, Joint.HEAD]  = positions[:, Joint.CHEST] + np.array([0, 0.25, 0])
    positions[:, Joint.L_HAND] = positions[:, Joint.L_WRIST]
    positions[:, Joint.R_HAND] = positions[:, Joint.R_WRIST]
    positions[:, Joint.L_FOOT] = positions[:, Joint.L_ANKLE] + np.array([0, -0.05, 0.05])
    positions[:, Joint.R_FOOT] = positions[:, Joint.R_ANKLE] + np.array([0, -0.05, 0.05])
    positions[:, Joint.L_TOE]  = positions[:, Joint.L_FOOT]  + np.array([0,  0,    0.1])
    positions[:, Joint.R_TOE]  = positions[:, Joint.R_FOOT]  + np.array([0,  0,    0.1])

    rotations = np.zeros((T, NUM_JOINTS, 4), dtype=np.float32)
    rotations[:, :, 0] = 1.0
    contacts = np.zeros((T, 4), dtype=np.float32)

    return NSFSequence(
        positions=positions, rotations=rotations, contacts=contacts,
        fps=fps, source="g1_body_positions", name="validation"
    )


def compute_metrics(pred: np.ndarray, gt: np.ndarray, dof_names: list[str]):
    """Per-joint and overall error metrics."""
    err = np.abs(pred - gt)   # (T, N)
    mae   = err.mean(axis=0)  # (N,)
    rmse  = np.sqrt((err**2).mean(axis=0))
    max_e = err.max(axis=0)

    print(f"\n{'Joint':<35} {'MAE(rad)':>9} {'RMSE(rad)':>10} {'Max(rad)':>9} {'MAE(deg)':>9}")
    print("-" * 75)
    for i, name in enumerate(dof_names):
        flag = " ←" if mae[i] > 0.3 else ""
        print(f"  {name:<33} {mae[i]:>9.4f} {rmse[i]:>10.4f} {max_e[i]:>9.4f} {np.degrees(mae[i]):>9.2f}{flag}")

    print("-" * 75)
    print(f"  {'OVERALL':<33} {mae.mean():>9.4f} {rmse.mean():>10.4f} {max_e.mean():>9.4f} {np.degrees(mae.mean()):>9.2f}")
    return mae, rmse, max_e


def plot_comparison(pred: np.ndarray, gt: np.ndarray, dof_names: list[str], fps: float):
    """Plot predicted vs ground truth for the worst joints."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    mae = np.abs(pred - gt).mean(axis=0)
    worst_idx = np.argsort(mae)[-6:][::-1]  # top 6 worst joints

    T = pred.shape[0]
    t = np.arange(T) / fps

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    fig.suptitle("Retargeting validation: predicted vs ground truth\n(6 worst joints by MAE)", fontsize=12)

    for ax, idx in zip(axes.flat, worst_idx):
        ax.plot(t, np.degrees(gt[:, idx]),   label="original",  lw=1.5, alpha=0.8)
        ax.plot(t, np.degrees(pred[:, idx]), label="retargeted", lw=1.5, alpha=0.8, linestyle="--")
        ax.set_title(f"{dof_names[idx]}  MAE={np.degrees(mae[idx]):.1f}°", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (deg)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "/tmp/retarget_validation.png"
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved: {out}")
    plt.show()


def main():
    # ── Load original data ──────────────────────────────────────────────────
    print(f"Loading: {NPZ_PATH}")
    d = np.load(NPZ_PATH)
    body_positions = d["body_positions"]    # (T, B, 3)
    body_names     = d["body_names"].tolist()
    dof_positions_gt = d["dof_positions"]  # (T, N) ground truth
    dof_names      = d["dof_names"].tolist()
    fps            = float(d["fps"])
    T              = body_positions.shape[0]

    print(f"  frames={T}  fps={fps}  DOFs={len(dof_names)}  bodies={len(body_names)}")
    print(f"  body_names: {body_names}")

    # ── Build NSF from body positions ────────────────────────────────────────
    print("\nBuilding NSF from body_positions...")
    seq = body_positions_to_nsf(body_positions, body_names, fps)
    print(f"  NSF: {seq.num_frames} frames, {len(BODY_TO_NSF)} joints mapped")

    # ── Run retargeter ───────────────────────────────────────────────────────
    print(f"\nRunning URDFRetargeter (URDF={URDF_PATH.name})...")
    retargeter = URDFRetargeter(URDF_PATH, MESH_DIR, robot_name="g1")
    result = retargeter.retarget(seq)
    dof_positions_pred = result["dof_positions"]  # (T, N)
    pred_names = result["dof_names"]

    # ── Align DOF ordering ───────────────────────────────────────────────────
    # retargeter may return DOFs in different order than NPZ
    gt_order   = {n: i for i, n in enumerate(dof_names)}
    pred_order = {n: i for i, n in enumerate(pred_names)}
    common = [n for n in dof_names if n in pred_order]

    gt_idx   = [gt_order[n]   for n in common]
    pred_idx = [pred_order[n] for n in common]

    gt_aligned   = dof_positions_gt[:, gt_idx]
    pred_aligned = dof_positions_pred[:, pred_idx]

    print(f"  Comparing {len(common)}/{len(dof_names)} DOFs")

    # ── Metrics ──────────────────────────────────────────────────────────────
    print("\n=== Per-joint error (retargeted vs original) ===")
    mae, rmse, max_e = compute_metrics(pred_aligned, gt_aligned, common)

    overall_mae_deg = np.degrees(mae.mean())
    print(f"\nSummary:")
    print(f"  Overall MAE : {overall_mae_deg:.2f} deg")
    if overall_mae_deg < 5:
        print("  Quality     : GOOD — retargeter is accurate")
    elif overall_mae_deg < 15:
        print("  Quality     : ACCEPTABLE — some joints need tuning")
    else:
        print("  Quality     : POOR — retargeter needs calibration")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_comparison(pred_aligned, gt_aligned, common, fps)


if __name__ == "__main__":
    main()
