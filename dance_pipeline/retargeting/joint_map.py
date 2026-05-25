"""
NSF joint → robot end-effector / body frame mapping.

Each entry defines which NSF joints determine the target position
for a robot body link used in IK.
"""

from ..nsf.format import Joint

# Maps robot body link name → NSF joint(s) that define its target position.
# When two joints are given, their midpoint is used.
G1_BODY_TARGETS: dict[str, tuple[Joint, ...]] = {
    "pelvis":              (Joint.ROOT,),
    "left_hip_yaw_link":   (Joint.L_HIP,),
    "right_hip_yaw_link":  (Joint.R_HIP,),
    "left_knee_link":      (Joint.L_KNEE,),
    "right_knee_link":     (Joint.R_KNEE,),
    "left_ankle_roll_link":(Joint.L_ANKLE,),
    "right_ankle_roll_link":(Joint.R_ANKLE,),
    "torso_link":          (Joint.CHEST,),
    "left_shoulder_pitch_link": (Joint.L_SHOULDER,),
    "right_shoulder_pitch_link":(Joint.R_SHOULDER,),
    "left_elbow_link":     (Joint.L_ELBOW,),
    "right_elbow_link":    (Joint.R_ELBOW,),
    "left_rubber_hand":    (Joint.L_WRIST,),
    "right_rubber_hand":   (Joint.R_WRIST,),
}

# Limb bone: (child_nsf_joint, parent_nsf_joint) → robot DOF link
# Used for proportional scaling.
G1_LIMB_SCALE_PAIRS: list[tuple[Joint, Joint, str]] = [
    (Joint.L_KNEE,    Joint.L_HIP,      "left_knee_link"),
    (Joint.R_KNEE,    Joint.R_HIP,      "right_knee_link"),
    (Joint.L_ANKLE,   Joint.L_KNEE,     "left_ankle_roll_link"),
    (Joint.R_ANKLE,   Joint.R_KNEE,     "right_ankle_roll_link"),
    (Joint.L_ELBOW,   Joint.L_SHOULDER, "left_elbow_link"),
    (Joint.R_ELBOW,   Joint.R_SHOULDER, "right_elbow_link"),
    (Joint.L_WRIST,   Joint.L_ELBOW,    "left_rubber_hand"),
    (Joint.R_WRIST,   Joint.R_ELBOW,    "right_rubber_hand"),
]
