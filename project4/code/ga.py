# Inherit from base class and do magical things :)
from nn import NN
import numpy as np
import random as rand

class GA(NN):
    def __init__(self, hidden_nodes='', mode='', training='', testing='',
        num_outputs='', pkl_file=''):
        super().__init__(hidden_nodes, mode, training, testing, num_outputs,
                       pkl_file)


    def train(self, pc, pm, num_chrom, tsk):
        # Train using the genetic algorithm
        # inputs
        # pc - Probability of crossover occuring
        # pm - probability of mutation occuring
        # num_chrom - number of individuals in the population
        # tsk - Tourniment selection number individuals selected

        population = []  # empty array to hold individuals in population
        # initilize population of size num_chrom
        for i in range(num_chrom):
            # get random weights and turn into chromosome then add to pop
            weights = self.initilize_weights()
            chromosome = self.weights_to_chromosome(weights)
            population.append(chromosome)

        # selection
        # tourniment selection, selecting k individuals
        tourniment = []
        select = np.random.permutation(len(population)-1)[0:tsk]
        # calculate the fitness for each selected individual
        best_fitness = float('inf')  # trying to minimize this value
        best_index = select[0]  # this will get changed
        for sel in select:
            weights = self.chromosome_to_weights(population[sel])
            fitness = self.calc_fitness(weights)
            #print(sel, fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_index = sel
        #print('best', best_index, best_fitness)
        
        # crossover and mutation
        # if random number is less than probability of crossover, do crossover
        if rand.uniform(0,1) <= pc:
            print('preforming crossover')


        # if random number less than probability of mutation, mutate with random
        # value
        if rand.uniform(0,1) <= pm:
            print('preforming mutation')

        # replacement



    def weights_to_chromosome(self, weights):
        # this method will convert the weight matrix for the nerual network
        # into a linear chormosome
        chromosome = []
        # for each layer in the weights array
        for layer in weights:
            # for each node in the layer
            for node in layer:
                # for each input to the node
                for inpu in node:
                    chromosome.append(inpu)
        return chromosome


    def chromosome_to_weights(self, chromosome):
        # Take a chromosome and turn it into a list of weight matricies for
        # use in the neural network

        # Nodes are rows and columns are inputs to those nodes
        # W[l]: (n[l], n[l-1]) (h, w)
        chrom_position = 0  # starting position on the chromosome for layer
        weights = []  # hold weight matricies for each layer
        for i in range(1, len(self.all_layers)):
            h = self.all_layers[i]   # number nodes in layer
            w = self.all_layers[i-1]   # number inputs to node
            
            # get all the weights in the layer matrix
            chrom_end_posit = chrom_position + h*w
            weight_list = chromosome[chrom_position:chrom_end_posit]
            # reshape into proper matrix
            weights.append(np.array(weight_list).reshape(-1, w))
            # set new chromosome starting position
            chrom_position = chrom_position + h*w

        return weights
