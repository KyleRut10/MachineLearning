# Inherit from base class and do magical things :)
from nn import NN
import numpy as np
import random as rand
import matplotlib.pyplot as plt

class DE(NN):
    def __init__(self, hidden_nodes='', mode='', training='', testing='',
        num_outputs='', pkl_file=''):
        super().__init__(hidden_nodes, mode, training, testing, num_outputs,
                       pkl_file)


    def train(self, num_chrom, max_generations=50, plot=False):
        # Train using the genetic algorithm
        # inputs
        # num_chrom - number of individuals in the population

        population = []  # empty array to hold individuals in population
        # initilize population of size num_chrom
        for i in range(num_chrom):
            # get random weights and turn into chromosome then add to pop
            weights = self.initilize_weights()
            chromosome = self.weights_to_chromosome(weights)
            population.append(chromosome)
    
        avg_fitness = self.calc_average_fitness(population)
        print('Generation 0 fitness: ', avg_fitness)
        avg_fitnesses = [avg_fitness]

        terminate = False
        generation = 0
        while not terminate:
            generation += 1

            offspring = []
            # for each memeber of the population
            for index,chrom in enumerate(population):
                # mutation
                trial = self.get_trial_vector(index, chrom, population)

                # recombination


                # replacement


                # evaluation
           

                break


            # final evalutation
            avg_fitness = self.calc_average_fitness(population)
            avg_fitnesses.append(avg_fitness)
            #print('Avg fitness: ', avg_fitness)
            
            # termination
            if generation >= max_generations:
                terminate = True
    
        print('Final avg. fitness gen ', generation, ': ', avg_fitness)
        self.training_fitnesses = avg_fitnesses
        if plot:
            self.plot_training()

    def calc_average_fitness(self, population):
        fitness = 0
        for chrom in population:
            weights = self.chromosome_to_weights(chrom)
            fitness += self.calc_fitness(weights)
        return fitness/len(population)
    

    def get_trial_vector(self, index, chrom, population):
        # generate three other chromosomes
        three = np.random.permutation(len(population))[0:3]
        # if the given chromosome is in there, keep trying until not
        while index in three:
            three = np.random.permutation(len(population))[0:3]
        # get the chromosomes from indexes
        x1 = population[three[0]]
        x2 = population[three[1]]
        x3 = population[three[2]]
        print(three)

        #return trial


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


    def plot_training(self):
        plt.plot(self.training_fitnesses, 'o')
        plt.xlabel('generations')
        plt.ylabel('error')
        plt.show()

