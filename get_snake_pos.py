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

# opening a socket on the server side
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
host = socket.gethostname()
port = 10500
s.bind((host, port))

# waiting until we recieve connection from the client (control script) 
s.listen(1)
connection, addr = s.accept()

#initialize some global variables
gravity_center = [0.0,0.0]
movement_vector = [0.0,0.0]

print "Got connection from", addr

# loading the pre-calculated weights file  for our neural network
dir = os.path.join(os.path.abspath('snn_stuff'),'snn_weights.npy')
W = np.load(dir)
from snn_stuff import GoalApproaching

# console menu function
def choose_data():
    global record_mode
    while not rospy.is_shutdown():
        user_input = raw_input(
"""Control of data collection:

    0:= Pause recording
    1:= Start recording

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

# normalization callback
# calculating all the vectors and sending them to control script
def callback_normalize(msg):
    global gravity_center
    global movement_vector
    global target_pos
    global time_passed
    global record_stopped
    global dir_vector
    global mov_vector
    global connection
    snake_tmp_pos = [0.0, 0.0]
    rotated_vector = [0.0, 0.0]

    if record_mode == 0:
        # just resetting the parameters
        time_passed = 0
        target_pos = [0.0, 0.0]

    if record_mode == 1:
        # we start our calculations by probing the target coordinates
        # in order to track the target we poll it's position constantly

        target_pos = [msg.pose[2].position.x, msg.pose[2].position.y]

        # counting our time
        time_passed += 1

        #short list of number meanings:
        # 800 = 1 sec
        # 400 = 0.5 sec
        # 160 = 200 msec
        # 40 = 50 msec
        if time_passed == 400:
            time_passed = 0
            target_relative_pos = [0.0,0.0]

            # starting to rotate the coordinates of a target
            target_relative_pos[0] = target_pos[0] - gravity_center[0]
            target_relative_pos[1] = target_pos[1] - gravity_center[1]

            vector_k = movement_vector

            normalisation_constant = math.sqrt(vector_k[0]**2 + vector_k[1]**2) # normalization constant for a movement vector

            vector_k[0] = vector_k[0]/normalisation_constant
            vector_k[1] = vector_k[1]/normalisation_constant

            y_changed = [0,1]

            # calculationg the rotation matrix
            Rotation_matrix = [[0,0],[0,0]]
            Rotation_matrix[0][0] = vector_k[1]
            Rotation_matrix[1][1] = vector_k[1]

            Rotation_matrix[0][1] = -vector_k[0]
            Rotation_matrix[1][0] = vector_k[0]

            # calculating the direction vector
            final_vector = [0, 0]

            final_vector[0] = Rotation_matrix[0][0] * target_relative_pos[0] + Rotation_matrix[0][1] * target_relative_pos[1]
            final_vector[1] = Rotation_matrix[1][0] * target_relative_pos[0] + Rotation_matrix[1][1] * target_relative_pos[1]

            normalisation_constant = math.sqrt(final_vector[0] ** 2 + final_vector[1] ** 2)
            final_vector = [final_vector[0] / normalisation_constant, final_vector[1] / normalisation_constant]

            # converting the direction vector in the format understood by the neural network
            result_array = [abs(final_vector[0]),abs(final_vector[1]),0,0]

            # sending the data to a neural network function to get the resulting turning pair of angles
            [angle0,angle1] = GoalApproaching.snn_testing(result_array,W)

            print('direction : ',result_array), '\n' # debug output
            print(angle0,angle1), '\n'

            # interpreting the neural network output
            if final_vector[1] >= 0: # figuring out the exact quadrant
                if final_vector[0] <= 0:
                    if abs(angle0)>24: #checking the threshold
                        movement = 1
                    else: movement = 0
                else:
                    if abs(angle0)<24: #checking the threshold
                        movement = 0
                    else: movement = 2
            else:
                if final_vector[0] >= 0:
                    movement = 2
                else:
                    movement = 1

            # sending the right movement to the controlling script
            connection.send(str(movement))

            #formal method solution - can fully replace the neural network solution

            '''
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
            '''

    return

# callback to get all the positions of links and calculate the snake movement vector
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

    # we need both rostopics because 'model_states' stores the target coordinates,
    # while 'link_states' stores the coordinates of all the snake link modules
    rospy.Subscriber("gazebo/model_states", ModelStates, callback_normalize)
    rospy.Subscriber("gazebo/link_states", LinkStates, callback_get_vector)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':

    # initializing global variables 
    global record_mode
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
    record_stopped = 0

	# setting up and running our console menu
    thread = Thread(target=choose_data)
    thread.start()
    # starting our listener subscriber node
    listener()
    thread.join()
