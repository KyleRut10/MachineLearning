import pandas as pd


def get_test_data():
    data = {}
    data['f1'] = [0,0,0,1,1,1,0,0,0]
    data['f2'] = [0,1,2,3,4,5,6,7,8]
    data['class'] = [0,0,0,1,1,1,0,0,0]

    df = pd.DataFrame.from_dict(data)
    return df
