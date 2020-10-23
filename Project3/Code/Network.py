

def network(inputs, num_layers, num_node, num_out, mode, momentum):
    # A multi-layer feedforward network with backpropogation.
    # Inputs -
    #     inputs: Arbitrary number of inputs to the algorithm
    #     num_layers: Number of Hidden Layers
    #     num_node: An x by num_layers array containing the
    #         number of hidden nodes per layer
    #     num_out: Arbitrary number of outputs from the algorithm
    #     mode: Whether the function is using performing regression
    #         or classification on the data.
    #     momentum: Whether or not/How much momentum should be
    #         used for the algorithm
    # Outputs -
    #     conv_rate: Convergence Rate of the algorithm
    #     F1_score: F1 score for the algorithm
    
    # Calculate Number of Inputs from inputs
    
    
    if(mode == "r"):
        # The first column of inputs should be the target
        return regression(inputs, num_layers, num_node, num_out)
    else if(mode == "c"):
        # The first row of inputs should be the class
        return classification(inputs, num_layers, num_node, num_out)
    
def regression(inputs, num_layers, num_node, num_out):
    # Regression by Multi-layer feedforward network with backpropogation
    # See network for inputs and outputs
    
    # Feedforward - An input pattern is applied to the input layer
    #     and it propogates layer by layer until an output is produced.
    #     Here this means we calculate an error term for the difference
    #     between the expected quantitative output and our output.
    
    # Backpropogation - Once the error
    
    # Don't forget momentum
