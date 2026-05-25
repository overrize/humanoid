#!/usr/bin/env python3
"""
Publish NPZ motion data as ROS JointState messages so RViz shows
the robot model moving.

Must be run with system Python (has rospy):
    /usr/bin/python3 -m dance_pipeline.rviz_publisher --npz motions/G1_dance.npz --urdf /tmp/g1_fixed.urdf

Or use the helper script:
    bash dance_pipeline/launch_rviz.sh motions/G1_dance.npz
"""

import sys
import argparse
import time
import numpy as np

# rospy lives in system Python
try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
except ImportError:
    print("ERROR: rospy not found. Run with: source /opt/ros/noetic/setup.bash && /usr/bin/python3 ...")
    sys.exit(1)


def load_npz(path: str):
    d = np.load(path)
    dof_names     = d["dof_names"].tolist()
    dof_positions = d["dof_positions"]   # (T, N)
    fps           = float(d["fps"])
    return dof_names, dof_positions, fps


def publish_motion(npz_path: str, loop: bool = True, speed: float = 1.0):
    dof_names, dof_positions, fps = load_npz(npz_path)
    T, N = dof_positions.shape

    rospy.init_node("dance_joint_publisher", anonymous=True)
    pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
    rate = rospy.Rate(fps * speed)

    print(f"Publishing {T} frames @ {fps:.0f} fps  ({T/fps:.1f}s)  loop={loop}")
    print(f"DOFs ({N}): {dof_names[:5]} ...")

    msg = JointState()
    msg.name     = dof_names
    msg.velocity = [0.0] * N
    msg.effort   = [0.0] * N

    while not rospy.is_shutdown():
        for t in range(T):
            if rospy.is_shutdown():
                break
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.position = dof_positions[t].tolist()
            pub.publish(msg)
            rate.sleep()
        if not loop:
            break

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",   required=True, help="NPZ motion file")
    parser.add_argument("--loop",  action="store_true", default=True)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args, _ = parser.parse_known_args()   # ignore ROS remapping args
    publish_motion(args.npz, loop=args.loop, speed=args.speed)


if __name__ == "__main__":
    main()
