

def network(inputs, num_layers, num_node, num_out, momentum, mode):
    # A multi-layer feedforward network with backpropogation.
    # Inputs -
    #     inputs: Arbitrary number of inputs to the algorithm
    #     num_layers: Number of Hidden Layers
    #     num_node: An x by num_layers array containing the
    #         number of hidden nodes per layer
    #     num_out: Arbitrary number of outputs from the algorithm
    #     momentum: Whether or not/How much momentum should be
    #         used for the algorithm
    #     mode: Whether the function is using performing regression
    #         or classification on the data.
    # Outputs -
    #     conv_rate: Convergence Rate of the algorithm
    #     F1_score: F1 score for the algorithm
    
    # Calculate Number of Inputs from inputs
    
    # Feedforward
    
    # Backpropogation
    
    # These differ in their error calculations predominantly, so the training
    #     should be the same
    if(mode == "r"):
        # 
    else if(mode == "c"):
        # The first row of inputs should be the class
        return classification(inputs, num_layers, num_node, num_out, momentum)
