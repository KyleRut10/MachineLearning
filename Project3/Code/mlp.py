import pandas as pd
import random as rand
import numpy as np


class MLP:
    # Class Variables
    # num_hidden: Number hidden layers
    # hidden_nodes: List of how many hidden nodes there are per layer
    # num_outputs: Number output nodes
    # train: df of training data
    # test: df of testing data
        
    # np vectorized version to compute sigmoid function on whole array of
    # values
    sigFunc = lambda t: 1/(1+np.exp(t))
    sig = np.vectorize(sigFunc)
    
    def __init__(self, hidden_nodes, num_outputs, training, test, mode):
        self.num_hidden = len(hidden_nodes)
        self.hidden_nodes = hidden_nodes
        self.num_outputs = num_outputs
        self.training = training
        self.test = test
        self.mode = mode
        # Make a cummulative list of how many nodes in each layer
        self.all_layers = [len(training.columns)-1]
        self.all_layers.extend(hidden_nodes)
        self.all_layers.append(num_outputs)
        self.layers = []
        self.layers.extend(hidden_nodes)
        self.layers.append(num_outputs)
        self.eda = 0.01

    def train(self):
        # Build weight matrices...
        self.weights = []
        
        # W[l]: (n[l], n[l-1]) (h, w)
        for i in range(1, len(self.all_layers)):
            h = self.all_layers[i]
            w = self.all_layers[i-1]
            hw = []
            for val in range(h*w):
                hw.append(rand.uniform(-0.01,0.01))
            # Reshape into 2D Numpy array
            self.weights.append(np.array(hw).reshape(h, w))

            #print(self.weights[-1])


        # TODO: Figure out how this is going to work
        converge = False
        while not converge:
            for i,pt in self.training.iterrows():
                print('pt: {} - {}'.format(i, list(pt)))
                activations = []
                # feedforward computation
                # initial inputs into first hidden layer
                # assuming class is in last position, so factor it out
                inputs = np.array(pt[:-1])
                activations.append(inputs)
                #print('Forward Propagation')
                for l,num_nodes in enumerate(range(len(self.layers))):
                    #print('layer: ', l)
                    # The weights going into layer l
                    W = self.weights[l]
                    z = np.matmul(W, inputs)
                    # compute activation function for whole layer
                    activations.append(self.sig(np.transpose(z)))
                    # update inputs into next layer
                    inputs = activations[-1]
                    #print(inputs)
            
                # backward propagation
                print('Backward propagation')
                # initiate weight update matrix and delta storage
                weight_updates = ['' for x in range(len(self.weights))]
                deltas = ['' for x in range(len(self.weights))]

                # calculate initial delta at output
                o_out = activations[-1]
                dj = pt[-1]  # target output of network for regression
                derr = -np.subtract(dj, o_out)
                do = np.matmul(o_out, np.subtract(1, o_out))
                # convert dnet to ndarray if it isn't
                if not isinstance(do, np.ndarray):
                    do = np.array([do])
                #print('derr: ', derr)
                #print('do: ', do)
                delta = np.matmul(derr, do)
                #print('delta: ', delta)
                # I think delta should be a scalar?
                if not isinstance(delta, np.ndarray):
                    delta = np.array([delta])
                print('delta out', delta)
                deltas[-1] = delta
                
                # TODO: I think this should be activations -2?
                dw = -np.matmul(activations[-1], delta) * self.eda
                if not isinstance(dw, np.ndarray):
                    dw = np.array([dw])
                weight_updates[-1] = dw
                print('output dw: ', weight_updates[-1])
                #print(weight_updates[-1])
                #print('activations')
                #for a in activations:
                #    print()
                #    print(a)

                #self.print_weights()
                # go back through hidden layers and update their weights
                # subtract 2, because already did the last position
                for i in range(len(self.weights)-2, -1, -1):
                    print('backprop layer: ', i)
                    oj = activations[i+1]  # outputs of layer
                    xj = activations[i][None, :]  # inputs to layer
                    wkj = self.weights[i+1]
                    derr = np.matmul(oj, np.subtract(1, oj))
                    if not isinstance(derr, np.ndarray):
                        derr = np.array([derr])[:, None]

                    # delta sum
                    print('weights', wkj.shape)
                    print('delta', delta.shape)
                    # IT'S GOING BY ROW!!!!
                    delta_sum = np.matmul(delta, wkj)
                    if len(delta_sum.shape) == 1:
                        delta_sum = delta_sum[None, :]
                    print('delta sum', delta_sum)
                    print('derr', derr.shape, 'sum', delta_sum.shape)
                    print('derr', derr)
                    delta = np.matmul(derr, delta_sum)
                    deltas[i] = delta
                    print('delta', delta)
                    
                    # calculate weight updates
                    print('xj', xj.shape, xj)
                    print('delta', delta.shape)
                    dw = -np.dot(np.transpose(delta), xj)
                    print('dw', dw.shape, dw)
                    weight_updates[i] = dw
                    #break

                ''' 
                # go back through hidden layers and update their weights
                # subtract 2, because already did the last position
                for i in range(len(self.weights)-2, -1, -1):
                    print('backprop layer: ', i)
                    oj = activations[i+1]  # outputs of layer
                    inputs = activations[i]
                    #print('previous delta: ', deltas[i+1])
                    # multiply delta by weight matrix
                    
                    print('deltas[i]:', deltas[i+1])
                    print('weights[i]:', self.weights[i])
                    #print(self.weights[i] * deltas[i+1])
                    #delta2 = np.matmul(self.weights[i], deltas[i+1])
                    #delta2 = np.matmul(deltas[i], self.weights[i])
                    #print('sum part: ', delta2)
                    #print('oj: ', oj)
                    do = np.matmul(oj, np.subtract(1, oj))
                    #print('do: ', do)
                    delta = do * delta2
                    #print('new delta: ', delta)
                    deltas[i] = delta

                    # calculate change in weights
                    #print('delta:', delta)
                    #print('newaxis:', delta[:, np.newaxis])
                    #dw = np.matmul(inputs, delta[:,np.newaxis()) #inputs * delta
                    print(delta)
                    dw = ''
                    #dw = delta * inputs

                    print('dw', dw)
                    #dw = -dw * self.eda
                    weight_updates[i] = dw
                    print('dw: ', dw)

                
                #print(weight_updates)
                # preform weight updates
                for i,w in enumerate(self.weights):
                    pass
                    #print('w', w.shape, 'wu', weight_updates[i].shape)
                    #self.weights[i] = np.add(w,weight_updates[i])
                #self.print_weights()
                '''
            break
    
    def print_weights(self):
        print('Weights')
        for i,w in enumerate(self.weights):
            print('layer ', i)
            print(w)
    
    def forward(self, W, a0):
        #z = np.matmult(W,a)
        #print(z)
        #print(1/(1+np.exp(z))
        pass
    
    def activation(self, z):
        pass

    def backward():
        pass

    def run():
        # Run one example through trained network
        pass
