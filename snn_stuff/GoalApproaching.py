import numpy as np
import math
from matplotlib import pyplot
import random
N = 25
WMAX = 3
C1 = 1/WMAX
EMAX = .8
EMIN = 0.2
MAX_STDP = 8
HIDDEN_LAYER = 4
INPUT_LAYER = 4
OUTPUT_LAYER = 2
EPOCHS = 100
EPMAX = EPOCHS # online calculation

TRAIN = False

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
    if t > 0: return 1
    else: return 0

def gij(w):
    return 1 - C1*w*math.exp(-1*abs(w)/WMAX)

def eta_val(epcurr):
    return EMAX - ((EMAX - EMIN)/EPMAX)*epcurr





class INeuron():
    def __init__(self):
        self.u = 0
        self.a = 0.2
        self.b = 0.025
        self.vth = 1
        self.tf = []

    def update(self,x,t):
        du = self.a * x + self.b  # assuming dt = 1
        if self.u >= self.vth:
            self.u = 0
            self.tf.append(t) # appending the time the spike occured
        else: self.u = self.u + du
        return

    def reset(self):
        self.tf = []
        self.u = 0

class HNeuron():
    def __init__(self):
        self.v = 0
        self.vth = 30
        self.tref = 3
        self.tm = 10
        self.gmax = 10
        self.ts = 5
        self.tf = []
        self.apre = np.zeros(shape=(INPUT_LAYER))
        self.delta_w = np.zeros(shape=(INPUT_LAYER)) # stores the changes associated with each hidden neuron
        self.apost = 0
        self.Apre = 0.4 # see if this needs to be changed
        self.Apost = 0.42 # see if this needs to be changed
        self.taupre = 10
        self.taupost = 10

    def reset(self):
        self.tf = []
        self.v = 0
        self.apre = np.zeros(shape=(INPUT_LAYER))
        self.apost = 0
        self.delta_w = np.zeros(shape=(INPUT_LAYER)) # stores the changes associated with each hidden neuron

    def alpha(self,t):
        return (self.gmax*t*math.exp(1 - t/self.ts))/self.ts

    def calc_stdp(self,dt): # change this during optimisation
        if dt >= 0: return 0.4*math.exp(-dt/10)
        else: return -0.42*math.exp(dt/10)

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
        stdp = 0
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
        self.v = 0
        self.vth = 25
        self.tref = 3
        self.tm = 10
        self.gmax = 10
        self.ts = 5
        self.tf = []
        self.apre = np.zeros(shape=(HIDDEN_LAYER))
        self.delta_w = np.zeros(shape=(HIDDEN_LAYER)) # stores the changes associated with each hidden neuron
        self.apost = 0
        self.Apre = .1 # see if this needs to be changed
        self.Apost = .105 # see if this needs to be changed
        self.taupre = 10
        self.taupost = 10



    def reset(self):
        self.v = 0
        self.tf = []
        self.apre = np.zeros(shape=(HIDDEN_LAYER))
        self.delta_w = np.zeros(shape=(HIDDEN_LAYER))  # stores the changes associated with each hidden neuron
        self.apost = 0

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
            self.v = 0
            self.tf.append(t)

    def calc_stdp(self,dt): # change this during optimisation
        if dt >= 0: return 0.4*math.exp(-dt/10)
        else: return -0.42*math.exp(dt/10)

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
            y = y + ((50 - tf)/50)*math.exp(0.05*(tf - t))
        return y

    def calculate_angle(self,t,index):
        if index == 0:
            return 90*self.encode_output(t)
        else:
            return -90*self.encode_output(t)

    def calculate_reward(self,y_data,t,index):
        return (abs(y_data) - abs(self.calculate_angle(t,index)))/110 # max angle = 90



class SNN():
    def __init__(self):
        self.T = 50
        self.frame = 0 # equivalent to dt - goes up to self.T
        random.seed(999)
        np.random.seed(999)
        hidden_weights = np.random.rand(OUTPUT_LAYER, HIDDEN_LAYER)*WMAX
        ip_weights = np.random.rand(HIDDEN_LAYER,INPUT_LAYER)*WMAX
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
        self.adj = 0.5

    def reset(self):
        for i in range(INPUT_LAYER):
            self.input_layer[i].reset()
        for i in range(HIDDEN_LAYER):
            self.hidden_layer[i].reset()
        for i in range(OUTPUT_LAYER):
            self.output_layer[i].reset()
        self.frame = 0
        return

    def process_input(self,x):
        '''
        :param x: non normalised input
        :return ip: 4 input values - x positive, y positive, x negative, y negative
        '''
        x = (x[0] / math.sqrt(x[0] ** 2 + x[1] ** 2), x[1] / math.sqrt(x[0] ** 2 + x[1] ** 2))
        ip = np.array([0.,0.,0.,0.])
        if x[0] >= 0:ip[0] = x[0]
        elif x[0] < 0: ip[2] = abs(x[0])

        if x[1] < 0: ip[3] = abs(x[1])
        elif x[1] >= 0: ip[1] = x[1]

        return ip

    def process_output(self,x):
        '''
        :param x: input in the 4*1 array form
        :return: output angle values - (angle pos,angle neg)
        '''
        op = np.zeros(shape=2)
        if x[3] > 0:
            if x[0] > 0: op[0],op[1] = 90, -180
            else: op[0],op[1] = 180,-90
        elif x[0] > 0: op[0],op[1] = np.clip(np.arcsin(x[0])*180/np.pi,10,90),-180
        elif x[2] > 0: op[0],op[1] = 180,-np.clip(np.arcsin(x[1])*180/np.pi,10,90)
        return op

    def gen_training_data(self,n):
        '''
        :param n: number of datapoints to be generated
        :return: data array consisting of n*4 values, first 3 being ip, next 2 being output
        '''
        print('Generating training data ...')
        data = np.zeros(shape=(n,6))
        for i in range(n):
            r1,r2 = random.randint(-50,50),random.randint(-50,50)
            if r1 == 0  and r2 == 0:
                r1, r2 = random.randint(-50, 50), random.randint(-50, 50)
            ip = self.process_input((r1,r2))
            op = self.process_output(ip)
            temp = np.append(ip,op)
            data[i] = temp
        print('data gen done.')
        return data

    def forward_pass(self,x_input,ep,train=False):
        tmat_input = []
        tmat_hidden = []
        print(x_input,self.frame)
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
                adjR = 0
        else:
            adjR = self.output_layer[1].calculate_reward(y_data[1],frame,index=1)
            if self.output_layer[0].calculate_angle(frame, 0) < 100:
                adjL = self.adj
            else:
                adjL = 0
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
        return






if __name__ == '__main__':
    snn = SNN()
    data = snn.gen_training_data(n=N)
    curr_ep = 0
    op = [[], []]
    errors = []
    accuracies = []
    if (TRAIN):
        print('training  ...............')
        for epoch in range(EPOCHS):
            for data_frame in data:
                for i in range(50):
                    tmat_input, tmat_hidden = snn.forward_pass(x_input=data_frame[0:4],ep=curr_ep,train=True)
                snn.backward_pass(frame=snn.frame,y_data=data_frame[4:6],tmat_input=tmat_input,tmat_hidden=tmat_hidden,curr_ep=curr_ep)
                snn.reset()
            curr_ep = curr_ep + 1
            '''every 10 epochs perform a training accuracy - number of correct turns/N'''
            if (1 == 1):
                accuracy = 0
                error = 0
                max_accuracy = 0
                for data_frame in data:
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
                        if max_accuracy < accuracy:
                            # save the model
                            W = snn.W
                    snn.reset()
                error = error / N
                errors.append(error)
                accuracy = (1-(accuracy / (80 * N)))*100 # max (max angle - min angle) percentage
                accuracies.append(accuracy)
                print('for epoch, error, accuracy : ',epoch,error,accuracy)

        dir = '/Users/karthikeyakaushik/Desktop/snn_temp'
        np.save(dir, W)
        pyplot.plot(range(0,len(errors)),errors)
        pyplot.show()
        pyplot.plot(range(0,len(accuracies)),accuracies)
        pyplot.show()
        print(W)
def snn_testing(data_frame, weights):
    snn = SNN()
    snn.W = weights
    #snn.reset()
    # data_frame - 4*1 vector
    curr_ep = 0
    for i in range(50):
        tmat_input, tmat_hidden = snn.forward_pass(x_input=data_frame, ep=curr_ep, train=False)
    print(snn.frame)
    print(tmat_input)
    op0 = snn.output_layer[0].calculate_angle(snn.frame, 0)
    op1 = snn.output_layer[1].calculate_angle(snn.frame, 1)
    print(op0, op1)
    return [op0,op1]








