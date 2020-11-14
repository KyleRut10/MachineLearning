# Inherit from base class and do magical things :)
from nn import NN

class GA(NN):
    def __init__(self, hidden_nodes='', mode='', training='', testing='',
        num_outputs='', pkl_file=''):
        super.__init__(hidden_nodes, mode, tarining, testing, num_outputs,
                       pkl_file)


    def train(self, pc, pm, num_chrom):
        # Train using the genetic algorithm
        # inputs
        # pc - Probability of crossover occuring
        # pm - probability of mutation occuring
        # num_chrom - number of individuals in the population

        population = []  # empty array to hold individuals in population
        # initilize population of size num_chrom
        for i in range(num_chrom):
            # get random weights and turn into chromosome then add to pop
            weights = self.initilize_weights()
            chromosome = self.weights_to_chromosome(weights)
            population.append(chromosome)

        # selection

        # crossover and mutation

        # replacement



    def weights_to_chromosome(self, weights):
        # this method will convert the weight matrix for the nerual network
        # into a linear chormosome

        pass
