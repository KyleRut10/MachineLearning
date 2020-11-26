# Inherit from base class and implement Particle swarm optimization
# this implementation is global best PSO
from nn import NN
import numpy as np
import random as rand

# position will be the weights from the neural net that we are trying to find with PSO
# velocity will be calculated in calc_v (current velocity) and be of the same structure as
    # position so it can be added component-wise to the position
# pbest is the personal best position of the particle as determined by the fitness function
# pbest_val is the fitness function value that pbest gave

class Particle():
    def __init__(self):
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_val = 10000 # some arbitrary big number

    def update_pos(self):
        self.position = (np.array(self.position) + np.array(self.velocity)).tolist()



# gbest is the best position in the whole swarm as determined by the fitness function
# gbest_val is the value returned from fitness function with input gbest
# population holds the particle objects in the swarm
class PSO(NN):
    def __init__(self, hidden_nodes='', mode='', training='', testing='',
                 num_outputs='', pkl_file=''):
        super().__init__(hidden_nodes, mode, training, testing, num_outputs,
                       pkl_file)
        self.gbest = None
        self.gbest_val = 10000 # some arbitrary big number
        self.population = []
        self.v_max = None
        self.omega = None
        self.cog_const = None
        self.social_const = None

    # Train using particle swarm optimization
    # v_max is the maximum velocity a particle can have (neg or pos) and is a hyperparameter
    # num_p is the number of particles in the swarm and is a hyperparameter
    # omega is the hyperparameter controlling the effect that inertia plays on the p velocity
    # cog_const is the hyperparameter in the velocity update equation for cognitive component
    # social_const is the hyperparameter in the velocity update equation for social component
    def train(self, num_p, v_max, omega, cog_const, social_const, max_iter):
        # initialize swarm
        self.omega = omega
        self.v_max = v_max
        self.cog_const = cog_const
        self.social_const = social_const
        for i in range(num_p):
            self.population.append(self.initialize_p())
        self.initialize_gbest()
        print("The average particle error before training:", self.find_avg_fitness())

        iterations = 0
        not_done = True
        while (not_done):
            #print("The average particle error while training:", self.find_avg_fitness())
            #print("The best particle error while training", self.gbest_val)
            self.check_bests()
            self.update_swarm()
            if (iterations > max_iter):
                not_done = False
            else:
                print(iterations)
                iterations += 1

        self.weights = self.chromosome_to_weights(self.gbest)
        print('Network error: ', self.calc_fitness(self.weights))
        print('The position was: ', self.gbest)


    # makes a particle with randomized positions
    def initialize_p(self):
        new_p = Particle()
        new_p.position = self.weights_to_chromosome(self.initilize_weights())
        v_arr = [] # making initial 0 velocity vector
        for i in new_p.position:
            v_arr.append(0)
        new_p.velocity = v_arr
        new_p.pbest = new_p.position.copy()
        new_p.pbest_val = self.calc_fitness(self.chromosome_to_weights(new_p.pbest))
        return new_p

    # sets up initial gbest
    def initialize_gbest(self):
        for p in self.population:
            fit_val = self.calc_fitness(self.chromosome_to_weights(p.position))
            if fit_val < self.gbest_val:
                self.gbest = p.position.copy()
                self.gbest_val = fit_val

    # check if an improvement has been made based on pbest_val vs the current position
    # as evaluated by the fitness function. Also checks and updates gbest
    def check_bests(self):
        for p in self.population:
            # get fitness value (which needs to be converted back to 2D weight vector first)
            fit_val = self.calc_fitness(self.chromosome_to_weights(p.position))
            if fit_val < p.pbest_val:
                p.pbest = p.position.copy()
                p.pbest_val = fit_val
                if fit_val < self.gbest_val:
                    self.gbest = p.position.copy()
                    self.gbest_val = fit_val

    def update_swarm(self):
        for p in self.population:
            p.velocity = self.compute_velocity(p)
            p.update_pos()

    # using numpy for component-wise calculations, then converts back
    # computes the new velocity of a single particle
    def compute_velocity(self, p):
        pos = np.array(p.position)
        vel = np.array(p.velocity)
        pb = np.array(p.pbest)
        gb = np.array(self.gbest)

        inertia = self.omega*vel
        cog = rand.uniform(0, self.cog_const)*(pb - pos)
        social = rand.uniform(0, self.social_const)*(gb - pos)
        new_vel = inertia + cog + social
        new_vel = new_vel.tolist()
        return self.clamp_velocity(new_vel)

    def clamp_velocity(self, vel):
        clamp_vel= []
        for v in vel:
            if (abs(v) > self.v_max):
                if (v > 0):
                    clamp_vel.append(self.v_max)
                else:
                    clamp_vel.append(-self.v_max)
            else:
                clamp_vel.append(v)

        return clamp_vel

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
                hw.append(rand.uniform(-2,2))
            # Reshape into 2D Numpy array
            weights.append(np.array(hw).reshape(h, w))
            #print(self.weights[-1])
            bias.append(np.array([0 for i in range(h)]).reshape(h, 1))
        return weights#, bias


    def find_avg_fitness(self):
        fit_sum = 0
        for p in self.population:
            fit_val = self.calc_fitness(self.chromosome_to_weights(p.position))
            fit_sum += fit_val
        return fit_sum/len(self.population)

# find avg velocity (abs value) of all the particles
    def find_avg_velocity(self):
        v_arr = np.array([])
        for i in self.population[0].velocity:
            v_arr = np.append(v_arr,0)
        for p in self.population:
            vel = [abs(v) for v in p.velocity]
            vel = np.array(vel)
            v_arr += vel
        return (v_arr/len(self.population)).tolist()




