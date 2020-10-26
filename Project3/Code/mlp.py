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
        self.all_layers = [len(training.columns)]
        self.all_layers.extend(hidden_nodes)
        self.all_layers.append(num_outputs)
        self.layers = []
        self.layers.extend(hidden_nodes)
        self.layers.append(num_outputs)
        

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
                activations = []
                # feedforward computation
                # initial inputs into first hidden layer
                inputs = np.array(pt)
                for l,num_nodes in enumerate(range(len(self.layers))):
                    print('layer: ', l)
                    # The weights going into layer l
                    W = self.weights[l]
                    z = np.matmul(W, inputs)
                    # compute activation function for whole layer
                    activations.append(self.sig(z))
                    # update inputs into next layer
                    inputs = activations[-1]
                    print(inputs)




            # NOTE: Keep weight updates in local variable, then put it in
            # self variable when do final updates

            break
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
