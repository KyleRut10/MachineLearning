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
        # select two parents
        num_parents = 2
        parents = []
        for i in range(num_parents):
            parent = self.best_tourniment_selection(tsk, population)
            parents.append(parent)

        # crossover and mutation
        # if random number is less than probability of crossover, do crossover
        # TODO: Make this more generalizable
        if rand.uniform(0,1) <= pc:
            parents = self.crossover(parent[0], parent2[1])

        # if random number less than probability of mutation, mutate with random
        # value
        for i,parent in enumerate(parents):
            if rand.uniform(0,1) <= pm:
                parents[i] = self.mutate_chromosome(parent)
        
        # replacement
         

    
    def worst_tourniment_selection(self, tsk, population):
        # tourniment selection, selecting k individuals
        tourniment = []
        select = np.random.permutation(len(population)-1)[0:tsk]
        # calculate the fitness for each selected individual
        worst_fitness = float('-inf')  # trying to maximize this value
        worst_index = select[0]  # this will get changed
        worst_chromosome = []
        for sel in select:
            weights = self.chromosome_to_weights(population[sel])
            fitness = self.calc_fitness(weights)
            #print(sel, fitness)
            if fitness > worst_fitness:
                worst_fitness = fitness
                worst_index = sel
                worst_chromosome = population[sel]
        #print('worst', worst_index, worst_fitness)
        return worst_index, worst_chromosome 
    

    def best_tourniment_selection(self, tsk, population):
        # tourniment selection, selecting k individuals
        tourniment = []
        select = np.random.permutation(len(population)-1)[0:tsk]
        # calculate the fitness for each selected individual
        best_fitness = float('inf')  # trying to minimize this value
        best_index = select[0]  # this will get changed
        best_chromosome = []
        for sel in select:
            weights = self.chromosome_to_weights(population[sel])
            fitness = self.calc_fitness(weights)
            #print(sel, fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_index = sel
                best_chromosome = population[sel]
        #print('best', best_index, best_fitness)
        return best_index, best_chromosome 
    

    def mutate_chromosome(self, chromosome):
        print('preforming mutation')
        # pick a random locus to mutate
        mutate_locus = rand.randrange(len(chromosome))
        # pick a random weight within the range of min and max weights
        min_w = min(chromosome)
        max_w = max(chromosome)
        #print('mutating locus: ', mutate_locus)
        #print(chromosome[mutate_locus])
        chromosome[mutate_locus] = rand.uniform(min_w, max_w)
        #print(chromosome[mutate_locus])
        return chromosome 


    def crossover(self, chrom1, chrom2):
        print('preforming crossover')
        # pick random locus to perform single point crossover at
        cross_point = rand.randrange(len(chrom1))
        # keep first half chrom 1
        new_chrom1 = chrom1[0:cross_point]
        # add chrom2 after cross point
        new_chrom1.extend(chrom2[cross_point:])
        # keep first half chrom 2
        new_chrom2 = chrom2[0:cross_point]
        # add crhom1 after cross point
        new_chrom2.extend(chrom1[cross_point:])
        return new_chrom1, new_chrom2


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
