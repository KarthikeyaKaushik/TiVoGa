#!/usr/bin/env python
import rospy
import math
import os.path
import socket
from math import pow
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Point
from threading import Thread
import numpy as np


s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
host = socket.gethostname()
port = 10500
s.bind((host, port))
s.listen(1)
connection, addr = s.accept()
gravity_center = [0.0,0.0]
movement_vector = [0.0,0.0]

print "Got connection from", addr
print "test"
dir = os.path.join(os.path.abspath('snn_stuff'),'snn_temp_testing.npy')
from snn_stuff import GoalApproaching
W = np.load(dir)
print "W : ",W

def choose_data():
    global record_mode
    while not rospy.is_shutdown():
        user_input = raw_input(
            """Control of data collection:
    0:= Pause recording
    1:= Start recording
    2:= Stop recording -- not implemented yet
    3:= Continue recording -- not implemented yet

    the new data is saved only if there is a change of position more than 10cm

Type 'exit' to quit.\n""")
        try:
            record_mode = int(user_input)
            if record_mode < 0 or record_mode > 1:
                record_mode = 0
            else:
                print "You entered:" + user_input
            return
        except ValueError:
            if user_input == 'exit':
                return
            print "Invalid Input"


def callback_normalize(msg):
    global gravity_center
    global movement_vector
    global target_pos
    global time_passed
    global record_stopped
    global target_captured
    global dir_vector
    global mov_vector
    global connection
    snake_tmp_pos = [0.0, 0.0]
    rotated_vector = [0.0, 0.0]

    if record_mode == 0:
        # just resetting the parameters
        target_captured = 0
        time_passed = 0
        target_pos = [0.0, 0.0]

    if record_mode == 1:
        # we start our calculations by probing the target coordinates
        #if we already have a target then we dont have to get this info again
        if target_captured == 0:
            target_pos = [msg.pose[2].position.x, msg.pose[2].position.y]
            target_captured = 1

        # counting our time
        time_passed += 1


        #short list of number meanings:
        # 800 = 1 sec
        # 400 = 0.5 sec
        # 160 = 200 msec
        # 40 = 50 msec
        if time_passed == 8:
            time_passed = 0
            target_relative_pos = [0.0,0.0]
            target_relative_pos[0] = target_pos[0] - gravity_center[0]
            target_relative_pos[1] = target_pos[1] - gravity_center[1]
            print "\n\n\n\n\n\n"
            print "Target\n"
            print target_relative_pos, '\n'

            vector_k = movement_vector
            print "Movement vector\n"
            print movement_vector, '\n'
            print "Center\n"
            print gravity_center, '\n'

            normalisation_constant = math.sqrt(vector_k[0]**2 + vector_k[1]**2)

            vector_k[0] = vector_k[0]/normalisation_constant
            vector_k[1] = vector_k[1]/normalisation_constant

            y_changed = [0,1]

            Rotation_matrix = [[0,0],[0,0]]
            Rotation_matrix[0][0] = vector_k[1]
            Rotation_matrix[1][1] = vector_k[1]

            Rotation_matrix[0][1] = -vector_k[0]
            Rotation_matrix[1][0] = vector_k[0]

            final_vector = [0, 0]

            final_vector[0] = Rotation_matrix[0][0] * target_relative_pos[0] + Rotation_matrix[0][1] * target_relative_pos[1]
            final_vector[1] = Rotation_matrix[1][0] * target_relative_pos[0] + Rotation_matrix[1][1] * target_relative_pos[1]

            normalisation_constant = math.sqrt(final_vector[0] ** 2 + final_vector[1] ** 2)
            final_vector = [final_vector[0] / normalisation_constant, final_vector[1] / normalisation_constant]


            #disabling the nn
            #don't delete
            '''
            result_array = [0,0,0,0]
            if final_vector[0] >= 0:
                result_array[0] = final_vector[0]
            else:
                result_array[2] = abs(final_vector[0])

            if final_vector[1] >= 0:
                result_array[1] = final_vector[1]
            else:
                result_array[3] = abs(final_vector[1])

            print('direction : ',result_array), '\n'
            [angle0,angle1] = GoalApproaching.snn_testing(result_array,W)
            print(angle0,angle1), '\n'
            angle0 = abs(angle0)
            angle1 = abs(angle1)
            movement = 0
            if angle0<angle1 and angle0>35:
                movement = 2
            elif angle1<angle0 and angle1>35:
                movement = 1
            '''
            #formal method solution
            print final_vector
            movement = 0
            if final_vector[1]>0:
                if final_vector[0]/final_vector[1]>0.6:
                    movement = 2
                elif final_vector[0]/final_vector[1]<-0.6:
                    movement = 1
            else:
                if final_vector[0]>0:
                    movement = 2
                else:
                    movement = 1
            connection.send(str(movement))

    return


def callback_get_vector(msg):
    global gravity_center
    global movement_vector
    if record_mode == 1:
        for i in range(1,18):
            gravity_center[0] += msg.pose[i].position.x
            gravity_center[1] += msg.pose[i].position.y
        gravity_center = [gravity_center[0]/17, gravity_center[1]/17]
        for i in range(1,9):
            movement_vector[0] += msg.pose[i].position.x - msg.pose[i+9].position.x
            movement_vector[1] += msg.pose[i].position.y - msg.pose[i+9].position.y
        movement_vector[0] = movement_vector[0] / 8.0
        movement_vector[1] = movement_vector[1] / 8.0
    return


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('get_pose', anonymous=True)

    # we need both rostopics because 'model_states' stores the head and 
    # target coordinates, while 'link_states' stores the coordinates of 
    # a tail module (as well as all the others intermediate modules)
    rospy.Subscriber("gazebo/model_states", ModelStates, callback_normalize)
    rospy.Subscriber("gazebo/link_states", LinkStates, callback_get_vector)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':

    # initializing global variables 
    # (mb should be moved to some __init__ section of a class and stopped from being global)
    global record_mode
    global target_captured
    global time_passed
    time_passed = 0.0
    global dir_vector
    dir_vector = [0.0, 0.0]
    global mov_vector
    mov_vector = [0.0, 0.0]
    global snake_head_pos
    snake_head_pos = [0.0, 0.0]
    global target_pos
    target_pos = [0.0, 0.0]
    global snake_tail_pos
    snake_tail_pos = [0.0, 0.0]
    global result_array
    result_array = [0.0, 0.0, 0.0, 0.0]

    mode_changed = 0
    record_mode = 0
    target_captured = 0
    record_stopped = 0

	# setting up and running our console menu
    thread = Thread(target=choose_data)
    thread.start()
	# opening the document to store our input - can just delete it if not needed
    listener()
    thread.join()
