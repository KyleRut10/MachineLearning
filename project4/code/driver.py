import pandas as pd
import data as dat
import mlp
import os
import itertools as it
import pickle
import nn


def get_example_dataset():
    data = {}
    data['f1'] = [1]
    data['f2'] = [4]
    data['f3'] = [5]
    data['regression'] = [1]
    df = pd.DataFrame.from_dict(data)
    return df
    

def get_test_data():
    data = {}
    data['f1'] = [.8,.7,.6,.5,.4,.3,.2,.1,0]
    data['f2'] = [0,.1,.2,.3,.4,.5,.6,.7,.8]
    data['class'] = [0,0,0,1,1,1,0,0,0]

    df = pd.DataFrame.from_dict(data)
    return df


def get_test_data2():
    data = {}
    data['f1'] = [0,0,0,1,1,1,0,0,0]
    data['f2'] = [0,1,2,3,4,5,6,7,8]
    data['class'] = [0,0,0,1,1,1,0,0,0]

    df = pd.DataFrame.from_dict(data)
    return df


def write_to_file_demo():
        aba = data.data_abalone()
        nna1 = mlp.MLP([3], aba, 'r')
        nna1.train(max_iterations=20)
        nna1_path = os.path.join('..', 'networks', 'nna1-test.pkl')
        nna1.save_network(nna1_path)

        nna1d = mlp.MLP(pkl_file=nna1_path)
        nna1d.train(max_iterations=5)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output) # pickle.HIGHEST_PROTOCOL)


def tuning(data, num_hl, mode, eda, max_iterations, filename):
    networks = []  # hold each trained network
    errors = []  # hold the errors for that network
    max_nodes = len(data.columns.values)+int(len(data.columns.values)*.01)
    min_nodes = int(len(data.columns.values)*.2)+2
    print('Tuning for {} hidden layers with {} max nodes per layer'.format(
          num_hl, max_nodes))
    print('Range: {} {}'.format(min_nodes, max_nodes))
    # make all possible permutations for each hidden layer
    combos = list(it.combinations_with_replacement(range(min_nodes,max_nodes), num_hl))
    # go through each combo
    for c in combos:
        print(c, end=" ")
        #print('Testing on combo: {}'.format(c))
        # create and train network
        nn = mlp.MLP(c, data, mode)
        nn.train(eda=eda, max_iterations=max_iterations) 
        # record the errors and network
        # round error to increase options
        errors.append(round(nn.training_statistics['training error'][-1],4)) 
        networks.append(nn)
        # save networks object to file so we can recreate if needed
        save_object(networks, filename)

    # print best results
    min_error = min(errors)
    print('The miminmum error was: {}'.format(min_error))
    best_results = []
    for i,e in enumerate(errors):
        if min_error == e:
            best_results.append(str(combos[i]))
    print('Min error for: {}'.format(', '.join(best_results)))  

if __name__ == '__main__':
    # example of calc fitness function
    ff = dat.data_forestfire()
    # 2 hidden layers with 3 nodes then 10
    # regression problem
    # Just going to use the whole dataset, but won't do this when doing actual
    # experiments
    nnf = nn.NN([3,10], 'r', ff)
    # get random weights
    weights = nnf.initilize_weights()
    # calculate the fitness for those weights
    error = nnf.calc_fitness(weights)
    print('Final error: ', error)
    '''
    iterations = 5000
    eda = 0.01
    ######## 0 Hidden Layers #######
    # Abalone dataset
    #print('\nAbalone')
    #data = dat.data_abalone()
    #nna = mlp.MLP([], data, 'r')
    #nna.train(max_iterations=iterations)
    #nna.save_network('../networks/aba.pkl')
    
    # Forestfire dataset
    #print('\nForestfire')
    #data = dat.data_forestfire()
    #nnf = mlp.MLP([], data, 'r')
    #nnf.train(max_iterations=iterations)
    #nnf.save_network('../networks/ff.pkl')
    
    # hardware
    #print('\nHardware')
    #data = dat.data_hardware()
    #nnh = mlp.MLP([], data, 'r')
    #nnh.train(max_iterations=iterations)
    #nnh.save_network('../networks/hw.pkl')
   
    # Glass
    print('\nGlass')
    data = dat.data_glass()
    nng = mlp.MLP([], data, 'c')
    nng.train(max_iterations=iterations)
    nng.save_network('../networks/g.pkl')

    # Breast Cancer
    print('\nBreast Cancer')
    data = dat.data_breast()
    nnb = mlp.MLP([], data, 'c')
    nnb.train(max_iterations=iterations)
    nnb.save_network('../networks/bx.pkl')

    # Soybean Small
    print('\nSoybean Small')
    data = dat.data_soybean_small()
    nns = mlp.MLP([], data, 'c')
    nns.train(max_iterations=iterations)
    nns.save_network('../networks/s.pkl')
   
    ####### 1 Hidden Layer
    print('One hidden layer')
    # Abalone dataset
    print('\nAbalone')
    data = dat.data_abalone()
    nna1 = mlp.MLP([10], data, 'r')
    nna1.train(max_iterations=iterations)
    nna1.save_network('../networks/aba-10.pkl')
    
    # Forestfire dataset
    print('\nForestfire')
    data = dat.data_forestfire()
    nnf1 = mlp.MLP([25], data, 'r')
    nnf1.train(max_iterations=iterations)
    nnf1.save_network('../networks/ff-25.pkl')
    
    # hardware
    print('\nHardware')
    data = dat.data_hardware()
    nnh1 = mlp.MLP([172], data, 'r')
    nnh1.train(max_iterations=iterations)
    nnh1.save_network('../networks/hw-172.pkl')
   
    # Glass
    #print('\nGlass')
    #data = dat.data_glass()
    #nng1 = mlp.MLP([4], data, 'c')
    #nng1.train(max_iterations=iterations)
    #nng1.save_network('../networks/g-4.pkl')

    # Breast Cancer
    print('\nBreast Cancer')
    data = dat.data_breast()
    nnb1 = mlp.MLP([20], data, 'c')
    nnb1.train(max_iterations=iterations)
    nnb1.save_network('../networks/b-20.pkl')

    # Soybean Small
    print('\nSoybean Small')
    data = dat.data_soybean_small()
    nns1 = mlp.MLP([43], data, 'c')
    nns1.train(max_iterations=iterations)
    nns1.save_network('../networks/s-43.pkl')
    
    ###### 2 Hidden Layers
    print('Two hidden layers')
    # Abalone dataset
    #print('\nAbalone')
    #data = dat.data_abalone()
    #nna2 = mlp.MLP([4,8], data, 'r')
    #nna2.train(max_iterations=iterations)
    #nna2.save_network('../networks/aba-4-8.pkl')
    
    # Forestfire dataset
    print('\nForestfire')
    data = dat.data_forestfire()
    nnf2 = mlp.MLP([18, 30], data, 'r')
    nnf2.train(max_iterations=iterations)
    nnf2.save_network('../networks/ff-18-30.pkl')
    
    # hardware
    print('\nHardware')
    data = dat.data_hardware()
    print()
    nnh2 = mlp.MLP([90,20], data, 'r')
    nnh2.train(max_iterations=iterations)
    nnh2.save_network('../networks/hw-90-20.pkl')
   
    # Glass
    print('\nGlass')
    data = dat.data_glass()
    nng2 = mlp.MLP([9,2], data, 'c')
    nng2.train(max_iterations=iterations)
    nng2.save_network('../networks/g-9-2.pkl')

    # Breast Cancer
    print('\nBreast Cancer')
    data = dat.data_breast()
    nnb2 = mlp.MLP([80,10], data, 'c')
    nnb2.train(max_iterations=iterations)
    nnb2.save_network('../networks/b-80-10.pkl')

    # Soybean Small
    print('\nSoybean Small')
    data = dat.data_soybean_small()
    nns2 = mlp.MLP([29,15], data, 'c')
    nns2.train(max_iterations=iterations)
    nns2.save_network('../networks/s-29-15.pkl')

    #results_path = os.path.join('..', 'networks', 'aba-results-1.pkl')
    #tuning(data, 1, 'r', eda, iterations, results_path)
    #print()
    #results_path = os.path.join('..', 'networks', 'aba-results-2.pkl')
    #tuning(data, 2, 'r', eda, iterations, results_path)
    
    # Forestfire dataset
    print('\nForestfire')
    data = data.data_forestfire()
    results_path = os.path.join('..', 'networks', 'ff-results-1.pkl')
    tuning(data, 1, 'r', eda, iterations, results_path)
    print()
    results_path = os.path.join('..', 'networks', 'ff-results-2.pkl')
    tuning(data, 2, 'r', eda, iterations, results_path)
    
    # hardware
    print('\nHardware')
    data = data.data_hardware()
    results_path = os.path.join('..', 'networks', 'hw-results-1.pkl')
    tuning(data, 1, 'r', eda, iterations, results_path)
    print()
    results_path = os.path.join('..', 'networks', 'hw-results-2.pkl')
    tuning(data, 2, 'r', eda, iterations, results_path)
   
    # Glass
    print('\nGlass')
    data = data.data_glass()
    results_path = os.path.join('..', 'networks', 'gl-results-1.pkl')
    tuning(data, 1, 'r', eda, iterations, results_path)
    print()
    results_path = os.path.join('..', 'networks', 'gl-results-2.pkl')
    tuning(data, 2, 'r', eda, iterations, results_path)

    # Breast Cancer
    print('\nBreast Cancer')
    data = data.data_breast()
    results_path = os.path.join('..', 'networks', 'br-results-1.pkl')
    tuning(data, 1, 'r', eda, iterations, results_path)
    print()
    results_path = os.path.join('..', 'networks', 'br-results-2.pkl')
    tuning(data, 2, 'r', eda, iterations, results_path)

    # Soybean Small
    print('\nSoybean Small')
    data = data.data_soybean_small()
    results_path = os.path.join('..', 'networks', 'sb-results-1.pkl')
    tuning(data, 1, 'r', eda, iterations, results_path)
    print()
    results_path = os.path.join('..', 'networks', 'sb-results-2.pkl')
    tuning(data, 2, 'r', eda, iterations, results_path)
    '''
