import numpy as np
import neural_network_setup
import pickle

def normalize_gradient(vector):
    return vector/np.linalg.norm(vector)

def back_propagation_multilayer(actual_table, network, num_epochs, lamb, activation_functions, activation_function_primes, test_network_accuracy, verbose):                    #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]          actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]       activation_functions = [layer_1_activation_function, layer_2_activation_function, ..., layer_n_activation_function] (same struction for activation_function_primes)
    epoch_count, accuracies_ls = 0, []
    while(epoch_count < num_epochs):
        if(verbose): print(f"EPOCH{epoch_count}:")
        for tup in actual_table:
            x, expected_output = tup
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = neural_network_setup.forward_propagate(x, network, activation_functions)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            #Backward Propagate - Calculate Gradient Descent Values
            delL_ls = [activation_function_primes[1](dot_vec)*(expected_output-a_vec)]                                              #del_Ls[-1] = delN (gradient function for LAST FUNCTION)
            for l in range(len(network)-2, -1, -1):
                delL_vector = activation_function_primes[l](dot_vecs_ls[l+1]) * ((network[l+1][0]).T @ delL_ls[0])
                delL_ls = [delL_vector] + delL_ls
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                #layer[1] = layer[1] + normalize_gradient(np.array([[lamb]]) * delL_ls[l])                                    #update bias
                #layer[0] = layer[0] + normalize_gradient(np.array([[lamb]]) * (delL_ls[l] @ (a_vecs_ls[l]).T))               #update weight
                layer[1] = layer[1] + np.array([[lamb]]) * delL_ls[l]                                   #update bias
                layer[0] = layer[0] + np.array([[lamb]]) * (delL_ls[l] @ (a_vecs_ls[l]).T)              #update weight

        accuracies_ls.append(test_network_accuracy(network, actual_table, activation_functions))
        if(verbose): print(f"Current Network Accuracy: {accuracies_ls[-1]}")
        epoch_count += 1
        lamb *= 0.99
    return network, accuracies_ls