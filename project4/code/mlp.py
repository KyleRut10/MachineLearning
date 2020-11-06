import pandas as pd
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pickle


class MLP:
    # Class Variables
    # num_hidden: Number hidden layers
    # hidden_nodes: List of how many hidden nodes there are per layer
    # num_outputs: Number output nodes
    # train: df of training data
    # test: df of testing data

    # np vectorized version to compute sigmoid function on whole array of
    # values
    sigFunc = lambda t: 1/(1+np.exp(-t))
    sig = np.vectorize(sigFunc)

    def __init__(self, hidden_nodes='', training='', mode='', num_outputs='',
        pkl_file=''):
        # read in saved network from file if file given
        if pkl_file != '':
            #print('Initilizing a network from a file')
            with open(pkl_file, 'rb') as f:
                obj = pickle.load(f)
            # make this object have all save values as one from file
            for key in obj.__dict__.keys():
                self.__dict__[key] = obj.__dict__[key]
            return
        #print('Initilizing a network from scratch')
        self.num_hidden = len(hidden_nodes)
        self.hidden_nodes = hidden_nodes
        self.training = training
        self.mode = mode.lower()
        # Get the number of output nodes based on the type, if it classifying,
        # pick the number of outputs based on the data's class column
        if self.mode == 'r':
            self.num_outputs = 1
        elif self.mode == 'c':
            self.num_outputs = len(training['class'].unique())
        else:
            print('***** MODE UNKNOWN *****')
            print('Will not work')
        if not isinstance(num_outputs, str):
            self.num_outputs = num_outputs
        # Make a cummulative list of how many nodes in each layer
        # Including inputs as layer
        self.all_layers = [len(training.columns)-1]  # -1 to remove class
        self.all_layers.extend(hidden_nodes)
        self.all_layers.append(self.num_outputs)
        # not including inputs as layer
        self.layers = []
        self.layers.extend(hidden_nodes)
        self.layers.append(self.num_outputs)
        self.training_statistics = {'status': 'not trained'}


    def train(self, eda=0.01, plot=False, max_iterations=5):
        #print('Training the network')
        # Hold the average training error for one round on dataset
        training_error = []
        # Build weight matrices...
        self.weights = []
        self.bias = []

        # Nodes are rows and columns are inputs to those nodes
        # W[l]: (n[l], n[l-1]) (h, w)
        for i in range(1, len(self.all_layers)):
            h = self.all_layers[i]   # number nodes in layer
            w = self.all_layers[i-1]   # number inputs to node
            hw = []
            for val in range(h*w):
                #hw.append(rand.uniform(-0.01,0.01))
                hw.append(rand.uniform(0,0.1))
            # Reshape into 2D Numpy array
            self.weights.append(np.array(hw).reshape(h, w))
            #print(self.weights[-1])
            self.bias.append(np.array([0 for i in range(h)]).reshape(h, 1))

        # Things to track iteration and convergance things
        converge = False
        # run no more than max_iterations times
        # max_iterations = 50 # input parameter
        # The difference between the sum of all the dw for the previous run
        # and the current run
        #max_dw_sum = 0.0001
        iteration = 0
        while not converge:
            # Keep track of average errors for each loop through the dataset
            iteration_error = []
            iteration += 1
            #print('*******Training iteration {}***********'.format(iteration))
            # save old weights
            old_weights = self.weights.copy()
            # randomize the dataset
            index = [i for i in range(self.training.shape[0])]
            rand.shuffle(index)
            randoms = self.training.set_index([index]).sort_index()
            # train the network on each point in the dataset
            for row_index,pt in self.training.iterrows():#randoms.iterrows():
                # compute all the activations in the feedforward step
                # activatioins[-1] is the final output of the network
                activations = self.feedforward(pt)

                #############################################################
                # backward propagation
                #############################################################
                ###print('Backward propagation')
                # initiate weight update matrix and delta storage
                weight_updates = ['' for x in range(len(self.weights))]
                deltas = ['' for x in range(len(self.weights))]

                # calculate initial delta at output
                o_out = activations[-1]   # output of the network
                # Get target output, different based on mode
                if self.mode.lower() == 'r':
                    d = pt[-1]  # target output of network for regression
                    d = np.array(d).reshape(1,1)
                else:
                    # get what the output should be
                    # Ex: if class = 2 from [1,2,3], this would return [0,1,0]
                    d = self.get_class_target(pt[-1])

                # calculate the inputs into the output layer from the output of
                # the previous layer and the weights
                x = np.add(np.matmul(self.weights[-1], activations[-2]), 
                           self.bias[-1])

                # Caclulate error
                if self.mode == 'r':
                    # calculate the squared error for regression
                    error = np.sum(np.subtract(d, o_out)**2)/len(o_out)
                    # calculate the intial delta at the output
                    delta = self.calc_delta_out_regres(o_out, d)
                else:
                    error = -np.log(o_out[int(pt[-1])][0])
                    # account for taking log of 0
                    if error < 0:
                        error = 0
                    delta = self.calc_delta_out_class(o_out, d)
                iteration_error.append(error)

                deltas[-1] = delta

                # calculate the weight changes
                #dw = -np.matmul(np.transpose(delta), x) * eda
                dw = -np.matmul(np.transpose(delta), x) * eda
                weight_updates[-1] = dw

                # go back through hidden layers and update their weights
                # subtract 2, because already did the last position
                for i in range(len(self.weights)-2, -1, -1):
                    #print('backprop layer: ', i)
                    o = activations[i+1]  # outputs of layer
                    x = activations[i]  # inputs to layer
                    w = self.weights[i+1]  # NOTE: why is it i+1????

                    # calculate a new delta
                    delta = self.calc_delta(o, deltas[i+1], w)
                    deltas[i] = delta

                    # calculate weight change for layer i
                    dw = -np.matmul(delta, np.transpose(x)) * eda
                    weight_updates[i] = dw

                # preform weight updates at the end
                for i,w in enumerate(self.weights):
                    self.weights[i] = np.subtract(w,weight_updates[i])
                #print(self.pweights())
            # average the error for the dataset round
            training_error.append(sum(iteration_error)/len(self.training))

            if iteration >= max_iterations:
                converge = True

        # print out the final average error
        print('Average last iteration error: {}'.format(training_error[-1]))

        # save the training error so can go back and anlyze later if needed
        self.record_statistics(eda, max_iterations, training_error, iteration)

        # make a plot of the error
        if plot:
            self.plot_error()

    def feedforward(self, pt):
        # will hold the calculated activations, the input to a layer
        # activations[0] is the input to the first hidden layer
        activations = []
        # feedforward computation
        # initial inputs into first hidden layer
        # assuming class is in last position, so factor it out
        activations.append(np.array(pt[:-1]).reshape(len(pt[:-1]), 1))

        for l in range(len(self.layers)):
            # The weights going into layer l
            W = self.weights[l]
            # compute activation function for whole layer
            z = np.matmul(W, activations[-1])  # input to layer
            # handle output layer based on regression or classification
            if l == len(self.layers)-1 and self.mode == 'c':
                # compute the softmax function at last node if classification
                # problem
                acts = np.exp(z)/np.sum(np.exp(z), axis=0)
            else:
                acts = self.sig(z)
            # convert to 2D numpy array
            activations.append(acts.reshape(len(acts), 1))

        return activations

    def run(self, pt):
        activations = self.feedforward(pt)
        # Caclulate error
        o_out = activations[-1]   # output of the network
        if self.mode == 'r':
            d = pt[-1]  # target output of network for regression
            d = np.array(d).reshape(1,1)
            # calculate the MSE for regression
            error = 0.5*np.sum(np.subtract(o_out, d)**2)/len(o_out)
        else:
            error = -np.log(o_out[int(pt[-1])][0])
            # account for taking log of 0
            if error < 0:
                error = 0
        return activations, error

    def calc_delta_out_regres(self, outputs, targets):
        # hold the values for each node's value of delta
        delta = []
        # Calculate deltaj for each output
        for j in range(len(outputs)):
            oj = outputs[j][0]  # singe output value
            dj = targets[j][0]  # single target value

            # caclulate derivatives
            delta.append(-(dj-oj)*oj*(1-oj))
        # convert to numpy array
        delta = np.array(delta).reshape(len(delta), 1)
        return delta

    def calc_delta_out_class(self, outputs, targets):
        # soft max derivative, cross entropy derivative
        delta = np.subtract(targets, outputs)
        return delta

    def calc_delta(self, outputs, delta_old, W):
        # hold each node's value for delta
        delta = []
        for j in range(len(outputs)):
            oj = outputs[j][0]  # Single output value

            # sum over each input into that node
            summ = 0
            for k in range(len(delta_old)):
                summ += delta_old[k] * W[k][j]

            delta.append(oj*(1-oj)*summ)
        delta = np.array(delta).reshape(len(delta), 1)
        return delta

    def get_class_target(self, class_val):
        # This will take in what numerical class this is and turn it into
        # an array, for example if the class is 1 out of 1,2,3, this will
        # return [1,0,0]
        d = []
        for i in range(self.num_outputs):
            if i == class_val:
               d.append(1)
            else:
                d.append(0)

        return np.array(d).reshape(self.num_outputs, 1)

    # write the object to a .pkl file so it can be read in later and the
    # same network can be reconstructed
    def save_network(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output) # pickle.HIGHEST_PROTOCOL)

    def pweights(self):
        print('Weights')
        for i,w in enumerate(self.weights):
            print('layer ', i+1, ' to ', i+2)
            print(w)

    def pactivations(self, activations):
        print('Activations (Outputs)')
        for i,act in enumerate(activations):
            if i == len(activations)-1:
                print('Output to network')
            else:
                print('Output to layer ', i+1)
            print(np.transpose(act))

    def record_statistics(self, eda, max_iterations, training_error,
        converged_in):
        ns = {'status': 'trained'}
        ns['eda'] = eda
        ns['max_iterations'] = max_iterations
        ns['training error'] = training_error
        ns['converged in'] = converged_in
        self.training_statistics = ns

    def print_statistics(self):
        pass

    def plot_error(self):
        plt.plot(self.training_statistics['training error'], 'o')
        plt.xlabel('iterations')
        plt.ylabel('error')
        plt.xlabel('iterations')
        plt.show()

    def print_network(self, activations):
        print('*** Layer 0***')
        print('Outputs per node')
        inputs = [str(round(i[0], 3)) for i in np.transpose(activations[0]).tolist()]
        print(', '.join(inputs))
        for i,a in enumerate(activations[1:]):
            print('***Layer {}***'.format(i+1))
            inputs = np.matmul(np.transpose(self.weights[i]), activations[i-1])
            inputs = np.transpose(inputs).tolist()
            inputs = [str(round(i[0], 3)) for i in inputs]
            print('Inputs per node')
            print(', '.join(inputs))
            outputs = [str(round(i[0], 3)) for i in activations[i]]
            print('Outputs per node')
            print(', '.join(outputs))
