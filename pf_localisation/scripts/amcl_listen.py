#!/usr/bin/env python3
import time
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

def log_to_file(fname, msgs, append=False):
    """
    If append = False, it will create a new log file with name fname and content msgs
    If append = True, it will append the messages (msgs) to the end of the file called fname

    :Args:
        | fname (str): the name of the file
        | msgs (list): list of messages
        | append (bool): if True new file, if False append at the end of the file 
    """
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(f'logs/{fname}', mode) as f:
        for msg in msgs:
            f.write(msg)

estimated_poses = []
odometry_poses = []
prev_odom_pose = "0 0 0"
time_init = time.time()

def callback(msg):
    global estimated_poses
    global odometry_poses
    global prev_odom_pose
    global time_init

    t = time.time() - time_init
    estimated_poses.append(f"{t} {msg.pose.pose.position.x} {msg.pose.pose.position.y} {msg.pose.pose.orientation.z}\n")
    log_to_file("amcl_estimated_pose.log", estimated_poses, append=False)
    
    odometry_poses.append(prev_odom_pose)
    log_to_file("amcl_odometry_pose.log", odometry_poses, append=False)

def odometry_callback(msg):
    global prev_odom_pose
    prev_odom_pose = f"{msg.pose.pose.position.x} {msg.pose.pose.position.y} {msg.pose.pose.orientation.z}\n"


def listener():
    rospy.init_node('amcl_listen', anonymous=True)
    rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, callback)
    rospy.Subscriber("odom", Odometry, odometry_callback, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    listener()