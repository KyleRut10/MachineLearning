import pandas as pd
import data
import mlp
import os
import itertools as it
import pickle


def get_test_data():
    data = {}
    data['f1'] = [8,7,6,5,4,3,2,1,0]
    data['f2'] = [0,1,2,3,4,5,6,7,8]
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
    max_nodes = len(data.columns.values)+int(len(data.columns.values)*.05)
    print('Tuning for {} hidden layers with max nodes per layer'.format(
          num_hl, max_nodes))
    # make all possible permutations for each hidden layer
    combos = list(it.combinations_with_replacement(range(1,max_nodes), num_hl))
    # go through each combo
    for c in combos:
        print('Testing on combo: {}'.format(c))
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
    # Abalone dataset
    data = data.data_abalone()
    results_path = os.path.join('..', 'networks', 'aba-results.pkl')
    tuning(data, 1, 'r', 0.01, 20, results_path)    
