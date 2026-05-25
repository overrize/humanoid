#!/bin/bash
# Launch RViz to preview a dance NPZ on the G1 robot model.
#
# Usage:
#   bash dance_pipeline/launch_rviz.sh [NPZ_FILE] [URDF_VARIANT]
#
# Examples:
#   bash dance_pipeline/launch_rviz.sh motions/G1_dance.npz
#   bash dance_pipeline/launch_rviz.sh motions/my_dance.npz g1_29dof_rev_1_0

set -e

NPZ="${1:-motions/G1_dance.npz}"
URDF_VARIANT="${2:-g1_29dof_rev_1_0}"

G1_DIR="/home/rexcon/unitree_ros/robots/g1_description"
ORIG_URDF="${G1_DIR}/${URDF_VARIANT}.urdf"
FIXED_URDF="/tmp/${URDF_VARIANT}_fixed.urdf"

source /opt/ros/noetic/setup.bash

# ── 1. Fix G1 URDF mesh paths (relative → absolute) ─────────────────────────
echo "[launch_rviz] Fixing URDF mesh paths..."
sed "s|filename=\"meshes/|filename=\"${G1_DIR}/meshes/|g" \
    "${ORIG_URDF}" > "${FIXED_URDF}"
echo "[launch_rviz] Fixed URDF: ${FIXED_URDF}"

# ── 2. Start roscore if not running ─────────────────────────────────────────
if ! rostopic list &>/dev/null; then
    echo "[launch_rviz] Starting roscore..."
    roscore &
    ROSCORE_PID=$!
    sleep 2
fi

# ── 3. Load URDF into ROS param ──────────────────────────────────────────────
echo "[launch_rviz] Loading robot_description..."
rosparam set /robot_description "$(cat ${FIXED_URDF})"

# ── 4. Start robot_state_publisher ──────────────────────────────────────────
echo "[launch_rviz] Starting robot_state_publisher..."
rosrun robot_state_publisher robot_state_publisher &
RSP_PID=$!

# ── 5. Start joint state publisher (our NPZ player) ─────────────────────────
echo "[launch_rviz] Starting joint publisher: ${NPZ}"
/usr/bin/python3 -m dance_pipeline.rviz_publisher --npz "${NPZ}" &
JSP_PID=$!

# ── 6. Open RViz ─────────────────────────────────────────────────────────────
RVIZ_CFG="/tmp/dance_preview.rviz"
cat > "${RVIZ_CFG}" << 'RVIZ_EOF'
Panels:
  - Class: rviz/Displays
    Name: Displays
Visualization Manager:
  Displays:
    - Class: rviz/RobotModel
      Name: RobotModel
      Enabled: true
      Robot Description: robot_description
      TF Prefix: ""
      Visual Enabled: true
      Collision Enabled: false
    - Class: rviz/TF
      Name: TF
      Enabled: false
  Global Options:
    Background Color: 30; 30; 30
    Fixed Frame: world
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 3.5
      Pitch: 0.3
      Yaw: 0.8
RVIZ_EOF

echo "[launch_rviz] Opening RViz..."
rviz -d "${RVIZ_CFG}"

# ── Cleanup ──────────────────────────────────────────────────────────────────
echo "[launch_rviz] Cleaning up..."
kill ${JSP_PID}  ${RSP_PID} 2>/dev/null || true
[ -n "${ROSCORE_PID}" ] && kill ${ROSCORE_PID} 2>/dev/null || true
