#!/usr/bin/env python
import numpy as np
import sys
import math
import os
from matplotlib import pyplot
import random
N = 90
WMAX = 20
WMAX_P = 30
WMAX_N = -25
C1 = 1.0/WMAX
EMAX = .2
EMIN = 0.05
MAX_STDP = 8.0
HIDDEN_LAYER = 4
INPUT_LAYER = 4
OUTPUT_LAYER = 2
EPOCHS = 100
EPMAX = EPOCHS # online calculation
SPLIT = 0 # data split for training on high accuracy requirement stuff

'''
DISCLAIMER : Comments do not necessarily reflect ground truths. Please forgive me. :p
'''

'''
Network architecture : 

4 input neurons - modelled in INeuron - vth = 1, a = 0.2, b = 0.025, with each neuron is associated the firing times - tf,
                  given by an array of values - x+,y+,x-,y-

20 hidden neurons - modelled in HNeuron - vth = 10, with each neuron is associated the firing times - tf,
                  given by an array of values

2 output neurons - modelled in ONeuron - vth = 15
'''

'''
Basic idea of training data generation, rewards etc.,

data generation :
[xpos,ypos,xneg,yneg,angle+,angle-] - angle+ is +x direction, angle- is -x direction
if yneg > 0: 
    if xpos > 0: angle+ = 90, angle- = -180
    if xneg > 0: angle- = -90, angle+ = 180
if ypos > 0:
    if xpos > 0: angle+ = arcsin(x), angle- = -180
    else: angle- = -arcsin(x), angle+ = 180 

training : 
backprop :
forward propagate the input for the entire time frame, T = 50ms,
to calculate rewards - 
    for the output synapses -
        consider the relevant output first - calculate reward as (ycon-ysnn)/ymax
        consider the irrelevant output next - calculate reward as shown in 5.4.2
    for the hidden synapses - 
        propagate the rewards back 

'''

def heavyside(t):
    if t > 0: return 1.0
    else: return 0.0

def gij(w):
    return 1 - C1*w*math.exp(-1*abs(w)/WMAX)

def eta_val(epcurr):
    return EMAX - ((EMAX - EMIN)/EPMAX)*epcurr





class INeuron():
    def __init__(self):
        self.u = 0.0
        self.a = 0.2
        self.b = 0.025
        self.vth = 1.0
        self.tf = []

    def update(self,x,t):
        du = self.a * x + self.b  # assuming dt = 1
        if self.u >= self.vth:
            self.u = 0.0
            self.tf.append(t) # appending the time the spike occured
        else: self.u = self.u + du
        return

    def reset(self):
        self.tf = []
        self.u = 0.0

class HNeuron():
    def __init__(self):
        self.v = 0.0
        self.vth = 30.0
        self.tref = 3.0
        self.tm = 10.0
        self.gmax = 10.0
        self.ts = 5.0
        self.tf = []
        self.apre = np.zeros(shape=(INPUT_LAYER))
        self.delta_w = np.zeros(shape=(INPUT_LAYER)) # stores the changes associated with each hidden neuron
        self.apost = 0.0
        self.Apre = 0.4 # see if this needs to be changed
        self.Apost = 0.41 # see if this needs to be changed
        self.taupre = 10.0
        self.taupost = 10.0

    def reset(self):
        self.tf = []
        self.v = 0.0
        self.apre = np.zeros(shape=(INPUT_LAYER))
        self.apost = 0.0
        self.delta_w = np.zeros(shape=(INPUT_LAYER)) # stores the changes associated with each hidden neuron

    def alpha(self,t):
        return (self.gmax*t*math.exp(1 - t/self.ts))/self.ts

    def calc_stdp(self,dt): # change this during optimisation
        if dt >= 0: return 0.4*math.exp(-dt/10.0)
        else: return -0.42*math.exp(dt/10.0)

    def stdp2(self,tmat,t,curr_ep,weights):
        '''
        :param tmat: input to the hidden neuron - the firing times of the input neurons.
        :param t: present time frame
        :param w: weights associated with that neuron in the backward direction
        :param weights: to calculate gij
        :return: none
        the eqns are updated according to 5.10,11,12 -
        Idea is that when pre synaptic neuron fires - ie., tmat[end] = present time, update self.apre, change self.delta[index]
        to reflect this change, and do the analogue when the post synaptic neuron fires.
        '''
        self.apost -= self.apost/self.taupost
        for index,n_tif in enumerate(tmat):
            self.apre[index] -= self.apre[index]/self.taupre
            if len(n_tif) > 0: # has to possess at least one pulse
                if n_tif[-1] == t: # implies a pre-synaptic pulse
                    self.apre[index] += self.Apre
                    self.delta_w[index] += self.apost*eta_val(curr_ep)*gij(weights[index]) # multiply with gij here
        if len(self.tf) > 0:
            if self.tf[-1] == t:
                self.apost -= self.Apost  # is this plus or minus? :(
                for index,_ in enumerate(tmat):
                    self.delta_w[index] += self.apre[index]*eta_val(curr_ep)*gij(weights[index]) # multiply with gij here


    def stdp(self,tmat):
        stdp = 0.0
        for tif in self.tf:
            for n_tif in tmat:
                for n_tf in n_tif:
                    stdp = stdp + self.calc_stdp(tif - n_tf)
        return stdp



    def psp(self,w,tmat,t,n_index): # w is only one row of the weight matrix representing connections between two layers.
        psp = 0 # t is the current time frame, n_index is the neuron index
        for counter,tif in enumerate(tmat): # tmat contains 3 lists, with each list representing firing times of neurons
            k = 0 # k = 2nd summation in 5.3
            for tf in tif:
                k = k + self.alpha(t - tf)*heavyside(t-tf)
            psp = psp + w[n_index][counter]*k
        return psp

    def update(self,w,tmat,t,n_index):
        if len(self.tf) > 0:
            if (t - self.tf[-1]) < self.tref: return# modelling the refractory period
        dv = (-self.v + self.psp(w,tmat,t,n_index))/self.tm
        if self.v < self.vth: self.v = self.v + dv
        else:
            self.v = 0
            self.tf.append(t)


class ONeuron():
    def __init__(self):
        self.v = 0.0
        self.vth = 30.0
        self.tref = 3.0
        self.tm = 10.0
        self.gmax = 10.0
        self.ts = 5.0
        self.tf = []
        self.apre = np.zeros(shape=(HIDDEN_LAYER))
        self.delta_w = np.zeros(shape=(HIDDEN_LAYER)) # stores the changes associated with each hidden neuron
        self.apost = 0.0
        self.Apre = .2 # see if this needs to be changed
        self.Apost = .21 # see if this needs to be changed
        self.taupre = 10.0
        self.taupost = 10.0



    def reset(self):
        self.v = 0.0
        self.tf = []
        self.apre = np.zeros(shape=(HIDDEN_LAYER))
        self.delta_w = np.zeros(shape=(HIDDEN_LAYER))  # stores the changes associated with each hidden neuron
        self.apost = 0.0

    def psp(self,w,tmat,t,n_index): # w is only one row of the weight matrix representing connections between two layers.
        psp = 0 # t is the current time frame
        for counter,tif in enumerate(tmat): # tmat contains 4 lists, with each list representing firing times of neurons
            k = 0 # k = 2nd summation in 5.3
            for tf in tif:
                k = k + self.alpha(t - tf)*heavyside(t-tf)
            psp = psp + w[n_index][counter]*k
        return psp

    def alpha(self,t):
        return (self.gmax*t*math.exp(1 - t/self.ts))/self.ts

    def update(self,w,tmat,t,n_index):
        if len(self.tf) > 0:
            if t - self.tf[-1] < self.tref : return
        dv = (-self.v + self.psp(w,tmat,t,n_index))/self.tm
        if self.v < self.vth: self.v = self.v + dv
        else:
            self.v = 0.0
            self.tf.append(t)

    def calc_stdp(self,dt): # change this during optimisation
        if dt >= 0: return 0.4*math.exp(-dt/10.0)
        else: return -0.42*math.exp(dt/10.0)

    def stdp2(self,tmat,t,curr_ep,weights):
        '''
        :param tmat: input to the hidden neuron - the firing times of the input neurons.
        :param t: present time frame
        :param weights: weights associated with that neuron in the backward direction
        :return: none
        the eqns are updated according to 5.10,11,12 -
        Idea is that when pre synaptic neuron fires - ie., tmat[end] = present time, update self.apre, change self.delta[index]
        to reflect this change, and do the analogue when the post synaptic neuron fires.
        '''
        self.apost -= self.apost/self.taupost
        for index,n_tif in enumerate(tmat):
            self.apre[index] -= self.apre[index]/self.taupre
            if len(n_tif) > 0: # has to possess at least one pulse
                if n_tif[-1] == t: # implies a pre-synaptic pulse
                    self.apre[index] += self.Apre
                    self.delta_w[index] += self.apost*eta_val(curr_ep)*gij(weights[index]) # multiply with gij here
        if len(self.tf) > 0:
            if self.tf[-1] == t:
                self.apost -= self.Apost  # is this plus or minus? :(
                for index,_ in enumerate(tmat):
                    self.delta_w[index] += self.apre[index]*eta_val(curr_ep)*gij(weights[index]) # multiply with gij here


    def stdp(self,tmat):
        stdp = 0
        for tif in self.tf:
            for n_tif in tmat:
                for n_tf in n_tif:
                    stdp = stdp + self.calc_stdp(tif - n_tf)
        return stdp

    def encode_output(self,t):
        y = 0
        for tf in self.tf:
            y = y + ((50 - tf)/50.0)*math.exp(0.05*(tf - t))
        return y

    def calculate_angle(self,t,index):
        if index == 0:
            return 90.0*self.encode_output(t)
        else:
            return -90.0*self.encode_output(t)

    def calculate_reward(self,y_data,t,index):
        return (abs(y_data) - abs(self.calculate_angle(t,index)))/110.0 # max angle = 90



class SNN():
    def __init__(self):
        self.T = 50
        self.frame = 0 # equivalent to dt - goes up to self.T
        hidden_weights = np.random.normal(20,3,[OUTPUT_LAYER, HIDDEN_LAYER])
        ip_weights = np.random.normal(4,1,[HIDDEN_LAYER,INPUT_LAYER])
        test = [ip_weights,hidden_weights]
        self.W = test
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        for i in range(INPUT_LAYER):
            self.input_layer.append(INeuron())
        for i in range(HIDDEN_LAYER):
            self.hidden_layer.append(HNeuron())
        for i in range(OUTPUT_LAYER):
            self.output_layer.append(ONeuron())
        self.adj = 1

    def reset(self):
        for i in range(INPUT_LAYER):
            self.input_layer[i].reset()
        for i in range(HIDDEN_LAYER):
            self.hidden_layer[i].reset()
        for i in range(OUTPUT_LAYER):
            self.output_layer[i].reset()
        self.frame = 0
        return

    def generate_training_data_circular(self,n):
        data = np.zeros(shape=(n,6))
        step_size = 90.0/N
        data_point = 0
        for i in np.arange(0.0, 90.0, step_size):
            x = np.cos(i * math.pi/180)
            y = np.sin(i * math.pi/180)
            input_vector = np.zeros(4)
            if x >= 0:
                input_vector[0] = x
            else:
                input_vector[2] = -x
            if y >= 0:
                input_vector[1] = y
            else:
                input_vector[3] = -y

            output_vector = np.zeros(2)
            if i<91:
                output_vector = np.array([90.0-i,-180.0])
            elif i<271:
                output_vector = np.array([180.0,90.0-i])
            else:
                output_vector = np.array([450.0-i,-180.0])


            data[data_point] = np.append(input_vector, output_vector)
            data_point += 1

        return data


    def generate_test_data(self,n):
        data_size = n / 10
        test_data = np.zeros(shape=(data_size,6))
        for data_point in range(data_size):
            i = random.randint(0,90)
            x = np.cos(i * math.pi/180)
            y = np.sin(i * math.pi/180)
            input_vector = np.zeros(4)
            if x >= 0:
                input_vector[0] = x
            else:
                input_vector[2] = -x
            if y >= 0:
                input_vector[1] = y
            else:
                input_vector[3] = -y

            output_vector = np.zeros(2)
            if i<91:
                output_vector = np.array([90.0-i,-180.0])
            elif i<271:
                output_vector = np.array([180.0,90.0-i])
            else:
                output_vector = np.array([450.0-i,-180.0])

            test_data[data_point] = np.append(input_vector, output_vector)

        return test_data



    def forward_pass(self,x_input,ep,train=False):
        tmat_input = []
        tmat_hidden = []
        for i in range(INPUT_LAYER):
            self.input_layer[i].update(x_input[i],self.frame)
            tmat_input.append(self.input_layer[i].tf)

        for i in range(HIDDEN_LAYER):
            self.hidden_layer[i].update(self.W[0],tmat_input,self.frame,i)
            tmat_hidden.append(self.hidden_layer[i].tf)

        for i in range(OUTPUT_LAYER):
            self.output_layer[i].update(self.W[1],tmat_hidden,self.frame,i)

        if train == True:
            for i in range(HIDDEN_LAYER):
                self.hidden_layer[i].stdp2(tmat_input,self.frame,ep,weights=self.W[0][i])

            for i in range(OUTPUT_LAYER):
                self.output_layer[i].stdp2(tmat_hidden,self.frame,ep,weights=self.W[1][i])

        self.frame = self.frame + 1
        return tmat_input,tmat_hidden

    def calc_adj(self,y_data,frame):
        if y_data[0] < 180:
            adjL = self.output_layer[0].calculate_reward(y_data[0],frame,index=0)
            if self.output_layer[1].calculate_angle(frame,1) > -100:
                adjR = self.adj
            else:
                adjR = 0.0
        else:
            adjR = self.output_layer[1].calculate_reward(y_data[1],frame,index=1)
            if self.output_layer[0].calculate_angle(frame, 0) < 100:
                adjL = self.adj
            else:
                adjL = 0.0
        return adjL,adjR


    def backward_pass(self,frame,y_data,tmat_input,tmat_hidden,curr_ep):
        '''
        first calculate rewards for all the synapes, then adjust synaptic weights according to eqn 5.6
        :param frame: time - 50 ms usually
        :param y_data: input data frame
        :return: none
        '''
        reward_op = np.zeros((OUTPUT_LAYER,HIDDEN_LAYER))
        [y0_reward,y1_reward] = self.calc_adj(y_data,frame)


        for i in range(HIDDEN_LAYER):
            reward_op[0][i] = y0_reward
            reward_op[1][i] = y1_reward

        reward_hidden = np.zeros((HIDDEN_LAYER,INPUT_LAYER))
        for counter,hidden_layer in enumerate(reward_hidden):
            reward = (self.W[1][0][counter]*reward_op[0][counter] + self.W[1][1][counter]*reward_op[1][counter])/(self.W[1][0][counter] + self.W[1][1][counter])
            reward_hidden[counter] = [reward,reward,reward,reward]
        total_reward = [reward_hidden,reward_op]
        weight_changes = [np.zeros((HIDDEN_LAYER,INPUT_LAYER)),np.zeros((OUTPUT_LAYER,HIDDEN_LAYER))]
        for counter1,hidden_neuron in enumerate(self.W[0]):
            for counter2,ip_neuron in enumerate(hidden_neuron):
                val_rij = total_reward[0][counter1][counter2]
                delta_w_2 = val_rij*self.hidden_layer[counter1].delta_w[counter2] # obtained using new stdp function
                weight_changes[0][counter1][counter2] = delta_w_2
        for counter1,output_neuron in enumerate(self.W[1]):
            for counter2, hidden_neuron in enumerate(output_neuron):
                val_rij = total_reward[1][counter1][counter2]
                delta_w_2 = val_rij*self.output_layer[counter1].delta_w[counter2] # obtained using new stdp function
                weight_changes[1][counter1][counter2] = delta_w_2
        for counter1,layer in enumerate(self.W):
            for counter2,sublayer in enumerate(layer):
                for counter3,weight in enumerate(sublayer):
                    self.W[counter1][counter2][counter3] = self.W[counter1][counter2][counter3] + weight_changes[counter1][counter2][counter3]
                    if self.W[counter1][counter2][counter3] > WMAX_P:
                        self.W[counter1][counter2][counter3] = WMAX_P
                    elif self.W[counter1][counter2][counter3] < WMAX_N:
                        self.W[counter1][counter2][counter3] = WMAX_N
        return



def snn_testing(data_frame, weights):
    snn = SNN()
    snn.W = weights
    snn.reset()
    # data_frame - 4*1 vector
    curr_ep = 0
    for i in range(50):
        _, _ = snn.forward_pass(x_input=data_frame, ep=curr_ep, train=False)
    op0 = snn.output_layer[0].calculate_angle(snn.frame, 0)
    op1 = snn.output_layer[1].calculate_angle(snn.frame, 1)
    return [op0,op1]



if __name__ == '__main__':
    snn = SNN()
    curr_ep = 0
    op = [[], []]
    errors = []
    accuracies = []
    # call with argument 0 for training
    if len(sys.argv)>1 and int(sys.argv[1]) == 0:
        random.seed(999)
        np.random.seed(999)
        data = snn.generate_training_data_circular(N)
        validation_data = snn.generate_test_data(N)
        print('training  ...............')
        W = snn.W
        for epoch in range(EPOCHS):
            for data_frame in data:
                for i in range(50):
                    tmat_input, tmat_hidden = snn.forward_pass(x_input=data_frame[0:4],ep=curr_ep,train=True)
                snn.backward_pass(frame=snn.frame,y_data=data_frame[4:6],tmat_input=tmat_input,tmat_hidden=tmat_hidden,curr_ep=curr_ep)
                snn.reset()
            curr_ep = curr_ep + 1
            '''every 10 epochs perform a training accuracy - number of correct turns/N'''
            max_accuracy = 0
            max_error = 0
            accuracy = 0.0
            error = 0.0
            for data_frame in validation_data:
                for i in range(50):
                    tmat_input, tmat_hidden = snn.forward_pass(x_input=data_frame[0:4], ep=curr_ep,train=False)
                op0 = snn.output_layer[0].calculate_angle(snn.frame, 0)
                op1 = snn.output_layer[1].calculate_angle(snn.frame, 1)
                turn_predicted = 1 * (abs(op0) > abs(op1)) # if turn is 0, angle in x pos, else if turn is 1, angle is xneg
                d_op0 = data_frame[4]
                d_op1 = data_frame[5]
                turn_actual = 1*(abs(d_op0) > abs(d_op1))
                if turn_predicted == turn_actual:
                    error = error + 1
                    if turn_predicted == 0:
                        accuracy = accuracy + abs((abs(op0) - abs(d_op0)))
                    else:
                        accuracy = accuracy + abs((abs(op1) - abs(d_op1)))
                snn.reset()
            error = error / N
            errors.append(error)
            accuracy = (1-(accuracy / (80 * N)))*100 # max (max angle - min angle) percentage
            snn.adj = 1 - error/100.0
            if max_accuracy < accuracy and max_error <= error:
                W = snn.W # saving the model
                max_error = error
                max_accuracy = accuracy
            accuracies.append(accuracy)
            print('for epoch, error, accuracy : ',epoch,error,accuracy)
        print(data)
        dir = os.path.abspath('snn_weights')
        np.save(dir, W)
        pyplot.plot(range(0,len(errors)),errors)
        pyplot.show()
        pyplot.plot(range(0,len(accuracies)),accuracies)
        pyplot.show()
        print(W)
    else:
        ip = [5,5]
        nor_constant = math.sqrt(ip[0]**2 + ip[1]**2)
        ip = [ip[0]/nor_constant,ip[1]/nor_constant]
        result_array = [0,0,0,0]
        if ip[0] >= 0: result_array[0] = ip[0]
        if ip[1] >= 0: result_array[1] = ip[1]
        if ip[0] < 0: result_array[2] = abs(ip[0])
        if ip[1] < 0: result_array[3] = abs(ip[1])
        print(result_array)
        weight_file = os.path.abspath('snn_weights.npy')
        print(snn_testing(result_array,np.load(weight_file)))









