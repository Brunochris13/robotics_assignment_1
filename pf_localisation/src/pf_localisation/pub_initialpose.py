#!/usr/bin/python3

import rospy
from geometry_msgs.msg import Pose, PoseWithCovariance, PoseWithCovarianceStamped


def talker():
    pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.init_node('talker', anonymous=True)

    pose = Pose()

    pose.position.x = .5
    pose.position.y = .2
    pose.position.z = 0

    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 1

    pose_cov = PoseWithCovariance()
    pose_cov.pose = pose

    pose_stamp = PoseWithCovarianceStamped()
    pose_stamp.header.frame_id = "map"
    pose_stamp.pose = pose_cov

    pub.publish(pose_stamp)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass