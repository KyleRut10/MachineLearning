import pandas as pd
import data
import mlp
import os


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


if __name__ == '__main__':
    aba = data.data_abalone()
    nna1 = mlp.MLP([3], aba, 'r')
    nna1.train(max_iterations=20)
    nna1_path = os.path.join('..', 'networks', 'nna1-test.pkl')
    nna1.save_network(nna1_path)

    nna1d = mlp.MLP(pkl_file=nna1_path)
    nna1d.train(max_iterations=5)
