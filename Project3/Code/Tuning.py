


def tune(inputs, num_layers, num_out, mode, momentum):
    # A function which outputs the number of nodes per hidden
    #     layer for the Backpropagation neural network
    # Inputs -
    #     inputs: Arbitrary number of inputs to the algorithm (col - 1)
    #     num_layers: Number of Hidden Layers
    #     num_out: Arbitrary number of outputs from the algorithm
    #     mode: Whether the function is using performing regression
    #         or classification on the data.
    #     momentum: Whether or not/How much momentum should be
    #         used for the algorithm
    # Output -
    #     num_node: An x by num_layers array containing the
    #         number of hidden nodes per layer
    
    # Calculate num_inputs from inputs
    
    # Here I am thinking we could create a list/dictionary for each
    #     of the layers then iterate the number of nodes for each
    #     layer one at a time in order to minimize F1-Score.
