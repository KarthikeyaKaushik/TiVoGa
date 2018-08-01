#!/usr/bin/env python
import rospy
import math
from math import pow
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Point
from threading import Thread
import numpy as np

dir = os.path.join(os.path.abspath('snn_stuff'),'snn_temp.npy')
from snn_stuff import GoalApproaching
W = np.load(dir)

s = socket.socket()
host = socket.gethostname()
port = 10500
s.bind((host, port))
s.listen(1)
connection, addr = s.accept()

print "Got connectione from", addr

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


def callback_calc_vector(msg):
    global snake_head_pos
    global snake_tail_pos
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
        snake_head_pos = [0.0, 0.0]

    if record_mode == 1:
        # we start our calculations by probing the target coordinates
        #if we already have a target then we dont have to get this info again
        if target_captured == 0:
            target_pos = [msg.pose[2].position.x, msg.pose[2].position.y]
            target_captured = 1

        # counting our time
        time_passed += 1

        # getting the current snake head coordinates and comparing them to stored ones
        #snake_tmp_pos[0] = msg.pose[1].position.x
        #snake_tmp_pos[1] = msg.pose[1].position.y
        #if abs(snake_tmp_pos[0] - snake_head_pos[0]) > 0.2 or abs(snake_tmp_pos[1] - snake_head_pos[1]) > 0.2:

        #short list of number meanings:
        # 800 = 1 sec
        # 400 = 0.5 sec
        # 160 = 200 msec
        # 40 = 50 msec
        if time_passed == 400:
            time_passed = 0
            snake_head_pos = [msg.pose[1].position.x, msg.pose[1].position.y]
            snake_tail_pos[0] = snake_tail_pos[0] - snake_head_pos[0]
            snake_tail_pos[1] = snake_tail_pos[1] - snake_head_pos[1]
            target_pos[0] = target_pos[0] - snake_head_pos[0]
            target_pos[1] = target_pos[1] - snake_head_pos[1]
            snake_head_pos = [0.0,0.0]

            vector_k = [0,0]

            vector_k[0] = snake_head_pos[0] - snake_tail_pos[0]
            vector_k[1] = snake_head_pos[1] - snake_tail_pos[1]

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

            final_vector[0] = Rotation_matrix[0][0] * target_pos[0] + Rotation_matrix[0][1] * target_pos[1]
            final_vector[1] = Rotation_matrix[1][0] * target_pos[0] + Rotation_matrix[1][1] * target_pos[1]

            normalisation_constant = math.sqrt(final_vector[0] ** 2 + final_vector[1] ** 2)
            final_vector = [final_vector[0] / normalisation_constant, final_vector[1] / normalisation_constant]

            result_array = [0,0,0,0]
            if final_vector[0] >= 0: result_array[0] = final_vector[0]
            if final_vector[1] >= 0: result_array[1] = final_vector[1]
            if final_vector[0] < 0: result_array[2] = abs(final_vector[0])
            if final_vector[1] < 0: result_array[3] = abs(final_vector[1])

            # calculating the head-target vector and general movement vector in "ground_plane" coordinates' system
         #    dir_vector[0] = (snake_head_pos[0] - target_pos[0]) / math.sqrt(
         #        pow(snake_head_pos[0] - target_pos[0], 2) + pow(snake_head_pos[1] - target_pos[1], 2))
         #    dir_vector[1] = (snake_head_pos[1] - target_pos[1]) / math.sqrt(
         #        pow(snake_head_pos[0] - target_pos[0], 2) + pow(snake_head_pos[1] - target_pos[1], 2))
         #    mov_vector[0] = (snake_head_pos[0] - snake_tail_pos[0]) / math.sqrt(
         #        pow(snake_head_pos[0] - snake_tail_pos[0], 2) + pow(snake_head_pos[1] - snake_tail_pos[1], 2))
         #    mov_vector[1] = (snake_head_pos[1] - snake_tail_pos[1]) / math.sqrt(
         #        pow(snake_head_pos[0] - snake_tail_pos[0], 2) + pow(snake_head_pos[1] - snake_tail_pos[1], 2))
         #    # rotating...
         #    rotated_vector[0] = mov_vector[0]*dir_vector[0] - mov_vector[1]*dir_vector[1]
         #    rotated_vector[1] = mov_vector[1]*dir_vector[0] + mov_vector[0]*dir_vector[1]
         #    # the result vector for the NN!
         #    result_array[0] = rotated_vector[0] if rotated_vector[0] > 0 else 0
         #    result_array[1] = rotated_vector[1]  if rotated_vector[1] > 0 else 0
         #    result_array[2] = rotated_vector[0] if rotated_vector[0] < 0 else 0
	    # result_array[3] = rotated_vector[1]  if rotated_vector[1] < 0 else 0
        [angle0,angle1] = GoalApproaching.snn_testing(W, result_array)
        print(angle0,angle1)
        # just writing result vector in file
        data.write("%s %s %s %s\n" % (result_array[0], result_array[1], result_array[2], result_array[3]))

       #print("the result array is (%f %f %f %f\n"%(result_array[0], result_array[1], result_array[2], result_array[3]))

        angle0 = abs(angle0)
        angle1 = abs(angle1)
        movement = 0
        if angle0<angle1 and angle0>30:
            movement = 1
        elif angle1<angle0 and angle1>30:
            movement = 2
        connection.send(str(movement))

    return


def callback_get_tail_coords(msg):
    global snake_tail_pos
    # just constantly (with ~200hz frequency) getting the tail coords
    if record_mode == 0:
        snake_tail_pos = [0.0, 0.0]
    if record_mode == 1:
        snake_tail_pos = [msg.pose[17].position.x, msg.pose[17].position.y]
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
    rospy.Subscriber("gazebo/model_states", ModelStates, callback_calc_vector)
    rospy.Subscriber("gazebo/link_states", LinkStates, callback_get_tail_coords)

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
    with open('position_data.txt', 'w') as data:
	# running the ros node
        listener()
    thread.join()
