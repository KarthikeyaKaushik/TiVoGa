#!/usr/bin/env python
"""
The node publishes joint position commands to effort position controllers.
The controllers should already be spawned and running and communicating
with the appropriate robot hardware interface.
"""

import rospy
import socket
import sys
from std_msgs.msg import Float64
from threading import Thread

import numpy as np


movement = 0
time_goal= 0
mode = 0

def read_movement():
    global movement
    global time_goal
    if mode == 0:
        input_message ="""Type to select a new movement. Possible values:
        0:=Slither Forward
        1:=Turn Left
        2:=Turn Right
        3:=Roll Left
        4:=Roll Right
        5:=Stop

    Type 'exit' to quit.\n"""
        while not rospy.is_shutdown():
            user_input = raw_input(input_message)
            try:
                new_movement = int(user_input)
                if new_movement<0 or new_movement>5:
                    new_movement = 5
                movement = new_movement
                if new_movement==1 or new_movement==2:
                    ### 150 = 30 degrees
                    time_goal = 150
            except ValueError:
                if user_input == 'exit':
                    sys.exit()
                print "Invalid Input"
    else:
        s = socket.socket()
        host = socket.gethostname()
        port = 10500
        s.connect((host,port))
        while not rospy.is_shutdown():
            new_movement = s.recv(8)
            try:
                new_movement = int(new_movement)
                if new_movement<0 or new_movement>5:
                    new_movement = 5
                movement = new_movement
                if new_movement==1 or new_movement==2:
                    ### 150 = 30 degrees
                    time_goal = 50
            except ValueError:
                raise ValueError('Movement must be an int')
        s.close()


class JointCmds:
    """
    The class provides a dictionary mapping joints to command values.
    """
    def __init__( self, num_mods):
        self.num_modules = num_mods
        self.jnt_cmd_dict = {}
        self.joints_list = []
        self.t = 0.0
        self.movement = 0
        for i in range(self.num_modules) :
            leg_str='S_'
            if i < 10 :
                leg_str += '0' + str(i)
            else :
                leg_str += str(i)
            self.joints_list += [leg_str]

    def update( self, dt, new_movement):
        self.t +=dt
        self.movement = new_movement
        global time_goal
        global movement

        ##0= Slither Forward
        if self.movement == 0:
            A_o = 80 * (np.pi/180)
            A_e = 50 * (np.pi/180)
            d_o = np.pi/3.0
            d_e = 2 * d_o
            w_o = 2
            w_e = 4


            for i,jnt in enumerate(self.joints_list):
                #odd
                if i%2:
                    P = ((i+1)/self.num_modules) * 0.3 + 0.7
                    theta = w_o * self.t + d_o * i
                    alpha = P * A_o * np.sin(theta)
                #even
                else:
                    P = ((i+1)/self.num_modules) * 0.7 + 0.3
                    theta = w_e * self.t + d_e * i
                    alpha = P * A_e * np.sin(theta)

                self.jnt_cmd_dict[jnt] = alpha
            return self.jnt_cmd_dict

        ##1:= Turn Left
        elif self.movement == 1:
            time_goal -= 1
            if time_goal == 0:
                movement = 0
            spat_freq = 0.25
            A = - np.pi/6.0
            d = 1
            for i, jnt in enumerate(self.joints_list) :
                if i<8:
                    self.jnt_cmd_dict[jnt] = A*np.sin( 2.0*np.pi*(d*self.t + i*spat_freq) )
                else:
                    self.jnt_cmd_dict[jnt] = -A*np.sin( 2.0*np.pi*(-d*self.t + i*spat_freq) )
            return self.jnt_cmd_dict


        ##2:= Turn Right 
        elif self.movement == 2:
            time_goal -= 1
            if time_goal == 0:
                movement = 0
            spat_freq = 0.25
            A = np.pi/6.0
            d = -1
            for i, jnt in enumerate(self.joints_list) :
                if i<8:
                    self.jnt_cmd_dict[jnt] = A*np.sin( 2.0*np.pi*(d*self.t + i*spat_freq) )
                else:
                    self.jnt_cmd_dict[jnt] = -A*np.sin( 2.0*np.pi*(-d*self.t + i*spat_freq) )
            return self.jnt_cmd_dict


        ##3:= Roll Left
        elif self.movement == 3:
            spat_freq = 0.25
            A = np.pi/6.0
            d = -1
            for i, jnt in enumerate(self.joints_list) :
                self.jnt_cmd_dict[jnt] = A*np.sin( 2.0*np.pi*(d*self.t + i*spat_freq) )
            return self.jnt_cmd_dict

        ##4:= Roll Right 
        elif self.movement == 4:
            spat_freq = 0.25
            A = - np.pi/6.0
            d = 1
            for i, jnt in enumerate(self.joints_list) :
                self.jnt_cmd_dict[jnt] = A*np.sin( 2.0*np.pi*(d*self.t + i*spat_freq) )
            return self.jnt_cmd_dict

        ##>=5:= Stop
        else:
            for i,jnt in enumerate(self.joints_list):
                self.jnt_cmd_dict[jnt] = 0
            return self.jnt_cmd_dict



def publish_commands( num_modules, hz ):
    pub={}
    ns_str = '/snake'
    cont_str = 'eff_pos_controller'
    for i in range(num_modules) :
        leg_str='S_'
        if i < 10 :
            leg_str += '0' + str(i)
        else :
            leg_str += str(i)
        pub[leg_str] = rospy.Publisher( ns_str + '/' + leg_str + '_'
                                        + cont_str + '/command',
                                        Float64, queue_size=10 )
    rospy.init_node('snake_controller', anonymous=True)
    rate = rospy.Rate(hz)
    jntcmds = JointCmds(num_mods=num_modules)
    thread = Thread(target=read_movement)
    thread.start()
    while not rospy.is_shutdown():
        jnt_cmd_dict = jntcmds.update(1./hz, movement)
        for jnt in jnt_cmd_dict.keys() :
            pub[jnt].publish( jnt_cmd_dict[jnt] )
        rate.sleep()
    thread.join()


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            mode = 0
        else:
            try:
                mode = int(sys.argv[1])
            except ValueError:
                sys.exit("Bad Argument")
            if mode != 0:
                mode = 1
        num_modules = 16
        hz = 100
        publish_commands( num_modules, hz )
    except rospy.ROSInterruptException:
        pass
