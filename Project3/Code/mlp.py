import pandas as pd
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import sys, os


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
        self.mode = mode.lower()
        # Make a cummulative list of how many nodes in each layer
        self.all_layers = [len(training.columns)-1]
        self.all_layers.extend(hidden_nodes)
        self.all_layers.append(num_outputs)
        self.layers = []
        self.layers.extend(hidden_nodes)
        self.layers.append(num_outputs)
        self.eda = .5#0.01

    def train(self):
        training_error = []
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
        
        
        # Things to track iteration and convergance things
        converge = False
        max_iterations = 15
        max_dw_sum = 0.0001
        iteration = 0
        while not converge:
            iteration_error = []
            iteration += 1
            #print('*******Training iteration {}***********'.format(iteration))
            # save old weights
            old_weights = self.weights.copy()
            for row_inex,pt in self.training.iterrows():
                activations = self.feedforward(pt)
                
                #############################################################
                # backward propagation
                #############################################################
                ###print('Backward propagation')
                # initiate weight update matrix and delta storage
                weight_updates = ['' for x in range(len(self.weights))]
                deltas = ['' for x in range(len(self.weights))]

                # calculate initial delta at output
                o_out = activations[-1]#[None, :]
                # difference for output measure
                if self.mode.lower() == 'r':
                    dj = pt[-1]  # target output of network for regression
                    dj = np.array(dj).reshape(1,1)
                else:
                    dj = self.get_class_target(pt[-1])
                xj = self.build_inputs(self.weights[-1], activations[-2])
                
                # TODO: Make this bariable later
                # Caclulate error
                iteration_error.append(0.5*np.sum(np.subtract(dj, o_out)**2))

                delta = self.calc_delta_out(o_out, dj)
                deltas[-1] = delta
                
                dw = -np.matmul(np.transpose(delta), xj) * self.eda
                # These can probably be delted
                #if not isinstance(dw, np.ndarray):
                #    dw = np.array([dw])
                #if len(dw.shape) == 1:
                #    dw = dw[None, :]
                weight_updates[-1] = dw
                
                # go back through hidden layers and update their weights
                # subtract 2, because already did the last position
                for i in range(len(self.weights)-2, -1, -1):
                    #print('backprop layer: ', i)
                    oj = activations[i+1]  # outputs of layer
                    xj = activations[i]#[None, :]  # inputs to layer
                    wkj = self.weights[i+1]
                    
                    # calculate a new delta
                    delta = self.calc_delta(oj, deltas[-1], wkj)
                    deltas[i] = delta
                    
                    dw = -np.matmul(delta, np.transpose(xj)) * self.eda
                    weight_updates[i] = dw
                
                # preform weight updates
                for i,w in enumerate(self.weights):
                    self.weights[i] = np.add(w,weight_updates[i])
                
            # average the error for the dataset round
            training_error.append(sum(iteration_error)/len(self.training))
                

            # Test for convergence by summing all the weights and seeing
            # if they are close to zero
            dw_sum_old = 0
            dw_sum_new = 0
            for i,old in enumerate(old_weights):
                dw_sum_old += np.sum(old)
                dw_sum_new += np.sum(self.weights[i])
            dw_diff = abs(dw_sum_new - dw_sum_old)
            if dw_diff < max_dw_sum*len(self.weights):
                converge = True
                print('Converged in {} iterations'.format(iteration))
            if iteration >= max_iterations:
                converge = True
                print('Max iterations ({}) reached, stopping'.format(iteration))
                print('old - new weights = {}'.format(dw_diff))

        '''
        plt.plot(training_error, 'o')
        plt.ylabel('error')
        plt.show()
        '''

    def feedforward(self, pt):
        # will hold the calculated activations, the input to a layer
        # activations[0] is the input to the first hidden layer
        activations = []
        # feedforward computation
        # initial inputs into first hidden layer
        # assuming class is in last position, so factor it out
        inputs = np.array(pt[:-1])
        inputs = inputs.reshape(len(inputs), 1)
        activations.append(inputs)
        
        for l,num_nodes in enumerate(range(len(self.layers))):
            # The weights going into layer l
            W = self.weights[l] #np.transpose(self.weights[l])
            # compute activation function for whole layer
            z = np.matmul(W, inputs)
            # handle output layer based on regression or classification
            if l == len(self.layers)-1 and self.mode == 'c':
                # compute the softmax function at last node if classification
                # problem
                acts = np.exp(z)/sum(np.exp(z))
            else:
                acts = self.sig(z)
            # convert to 2D numpy array
            acts = acts.reshape(len(acts), 1)
            activations.append(acts)
            # update inputs into next layer
            inputs = self.build_inputs(W, activations[-2])
       
            
        
        return activations 

    def calc_delta_out(self, outputs, targets):
        results = []
        for j in range(len(outputs)):
            oj = outputs[j][0]
            dj = targets[j][0]

            results.append(-(dj-oj)*oj*(1-oj))
        results = np.array(results)
        return results

    def calc_delta(self, outputs, delta_old, W):
        #print('CALCULATING DELTA')
        results = []
        for j in range(len(outputs)):
            #print('for node: ', j)
            oj = outputs[j][0]
            
            summ = 0
            for k in range(len(delta_old)):
                summ += delta_old[k] * W[k][j]

            results.append(oj*(1-oj)*summ)
        results = np.array(results).reshape(len(results), 1)
        return results
    
    def get_class_target(self, class_val):
        dj = []
        for i in range(self.num_outputs):
            if i == class_val:
               dj.append(1)
            else:
                dj.append(0)

        return np.array(dj).reshape(self.num_outputs, 1)
    
    def build_inputs(self, W, activation):
        activation = np.transpose(activation)[0]
        inputs = []
        for node in range(len(W)):
            ins = []
            for i,w in enumerate(W[node]):
                ins.append(w*activation[i])
            inputs.append(sum(ins))
        inputs = np.array(inputs).reshape(len(inputs), 1)
        return inputs

    def print_weights(self):
        print('Weights')
        for i,w in enumerate(self.weights):
            print('layer ', i)
            print(w)
    '''
    def softmax(self, W, X):
        # TODO: Write softmax function that takes
        
        return 1
    '''
