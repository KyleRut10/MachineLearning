

def network(inputs, num_layers, num_node, num_out, mode):
    # A multi-layer feedforward network with backpropogation.
    # Inputs -
    #     inputs: Arbitrary number of inputs to the algorithm
    #     num_layers: Number of Hidden Layers
    #     num_node: An x by num_layers array containing the
    #         number of hidden nodes per layer
    #     num_out: Arbitrary number of outputs from the algorithm
    #     mode: Whether the function is using performing regression
    #         or classification on the data.
    # Output(s?) -
    #     
    
    # Calculate Number of Inputs from inputs
    if(mode == "r"):
        return regression(inputs, num_layers, num_node, num_out)
    else if(mode == "c"):
        return classification(inputs, num_layers, num_node, num_out)
    
def regression(inputs, num_layers, num_node, num_out):
    # Regression by Multi-layer feedforward network with backpropogation
    # See network for inputs and outputs
    
    # Feedforward - An input pattern is applied to the input layer
    #     and it propogates layer by layer
