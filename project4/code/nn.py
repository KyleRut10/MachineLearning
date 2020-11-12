import numpy as np
import pandas as pd
# The base code will go here
# Idea: remove it from mlp and have mlp inherit from this class

class NN:
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

    def __init__(self, hidden_nodes='', mode='', training='', testing='',
        num_outputs='', pkl_file=''):
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
        self.testing = testing
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
        #self.num_outputs = 2 # NOTE: doing this fortesting
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


    def initilize_weights(self):
        # Build weight matrices
        weights = []
        bias = []

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
            weights.append(np.array(hw).reshape(h, w))
            #print(self.weights[-1])
            bias.append(np.array([0 for i in range(h)]).reshape(h, 1))
        return weights#, bias

    def calc_fitness(self, ea_weights):
        # calculate error for each point in the dataset
        training_error=[]

        for index,pt in self.training.iterrows():
            acts, error = self.run(pt, ea_weights)
            training_error.append(error)

        # take the average error
        return np.sum(training_error)/len(training_error)


    def feedforward(self, pt, weights=''):
        if isinstance(weights, str):
            weights = self.weights()
        # will hold the calculated activations, the input to a layer
        # activations[0] is the input to the first hidden layer
        activations = []
        # feedforward computation
        # initial inputs into first hidden layer
        # assuming class is in last position, so factor it out
        activations.append(np.array(pt[:-1]).reshape(len(pt[:-1]), 1))
        
        for l in range(len(self.layers)):
            # The weights going into layer l
            W = weights[l]
            # compute input to layer z
            z = np.matmul(W, activations[-1])
            # handle output layer based on regression or classification
            if l == len(self.layers)-1 and self.mode == 'c':
                # compute the softmax function at last node if classification
                # problem
                acts = np.exp(z)/np.sum(np.exp(z), axis=0)
                if np.isnan(np.sum(acts)):
                    import ipdb; ipdb.set_trace()
            else:
                acts = self.sig(z)
            #import ipdb; ipdb.set_trace()
            # convert to 2D numpy array
            activations.append(acts.reshape(len(acts), 1))

        return activations


    def run(self, pt, weights=''):
        if isinstance(weights, str):
            weights = self.weights()
        activations = self.feedforward(pt, weights)
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
            print('bias')
            print(self.bias[i])


    def pactivations(self, activations):
        print('Activations (Outputs)')
        for i,act in enumerate(activations):
            if i == len(activations)-1:
                print('Output to network')
            else:
                print('Output to layer ', i+1)
            print(np.transpose(act))

