# Inherit from base class and do magical things :)
from nn import NN

class GA(NN):
    def __init__(self, hidden_nodes='', mode='', training='', testing='',
        num_outputs='', pkl_file=''):
        super.__init__(hidden_nodes, mode, tarining, testing, num_outputs,
                       pkl_file)


    def train():
       pass 
