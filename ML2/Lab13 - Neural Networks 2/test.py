import numpy as np
import random
import math

def print_network(network):
    for l in range(len(network)):
        print(f"layer: {l}\nweight_matrix:\n{network[l][0]}\nbias_matrix:\n{network[l][1]}\n")

def create_random_wm(layer1_size, layer2_size):            #returns layer2_size x layer1_size matrix
    ls = []
    for row in range(layer2_size): ls.append([random.uniform(-1, 1) for i in range(layer1_size)])
    return np.array(ls)

def create_random_bs(layer_size): return np.array([[random.uniform(-1, 1) for i in range(layer_size)]]).T       #returns layer_size x 1

def create_network(layer_sizes):        #layer_sizes = [input_layer_size, layer1_size, layer2_size, ... , output_layer_size]
    network = []
    for i in range(1, len(layer_sizes)):
        size1 = layer_sizes[i-1] 
        size2 = layer_sizes[i]                      #current layer size
        wm, bs = create_random_wm(size1, size2), create_random_bs(size2)
        network.append([wm, bs])
    return network


def sigmoid(dot):                       #input: np.array(); returns a np.array() w/ sigmoid function applied to each entry
    arr = [(1/(1+math.exp(-dot[i][0]))) for i in range(dot.shape[0])]
    return np.array([arr]).T

def sigmoid_prime(dot):
    return (sig:=sigmoid(dot)) * (1-sig)

def find_dot(w, b, x):                            #@ is normal matrix multiplication
    return (w@x) + b

def perceptron(A, dot): return A(dot)           #returns f* = A(w dot x + b)<np.array>

def forward_propagate(x, network, activation_function):                      #propagates through network and returns dot_vecs_ls and a_vecs_ls of; values of each layer
    a_vec = x
    dot_vecs_ls, a_vecs_ls = [np.array([[0, 0]])], [a_vec]         #don't use dot_vec[0]; just a placeholder
    for layer in network:
        w_matrix, b_scalar = layer[0], layer[1]
        dot_vec = find_dot(w_matrix, b_scalar, a_vec)
        a_vec = perceptron(activation_function, dot_vec)
        dot_vecs_ls.append(dot_vec)
        a_vecs_ls.append(a_vec)
    return a_vec, dot_vec, dot_vecs_ls, a_vecs_ls

def test_network_accuracy(network, actual_table, activation_function):
    num_correct, num_instances, misclassified_ls = 0, len(actual_table), []
    for tup in actual_table:
        x, expected_out = tup
        #print(f"Input: \n{x}")
        #print(f"Expected_out: {expected_out}")
        #print(f"Prediction: {forward_propagate(x, network, activation_function)[0]}")
        if(forward_propagate(x, network, activation_function)[0] == expected_out): num_correct += 1
        else: misclassified_ls.append(tup)
        #print(f"Current Accuracy: {(num_correct/num_instances)}")
        #input()
    return (num_correct/num_instances), misclassified_ls

def sigmoid_mnist(dot):                       #input: np.array(); returnsa np.array() w/ sigmoid function applied to each entry
    arr = []
    for i in range(dot.size): arr.append(1/(1+math.exp(-dot[0][i])))
    return np.array([arr])

def sigmoid_prime_mnist(dot):
    return (sig:=sigmoid_mnist(dot)) * (1-sig)

def find_dot_mnist(w, b, x): return (x@w) + b

def perceptron_mnist(A, dot): return A(dot)           #returns f* = A(w dot x + b)<np.array>

def forward_propagate_mnist(x, network, activation_function):                      #propagates through network and returns dot_vecs_ls and a_vecs_ls of; values of each layer
    a_vec = x
    dot_vecs_ls, a_vecs_ls = [np.array([[0, 0]])], [a_vec]         #don't use dot_vec[0]; just a placeholder
    for layer in network:
        w_matrix, b_scalar = layer[0], layer[1]
        dot_vec = find_dot_mnist(w_matrix, b_scalar, a_vec)
        a_vec = perceptron_mnist(activation_function, dot_vec)
        dot_vecs_ls.append(dot_vec)
        a_vecs_ls.append(a_vec)
    return a_vec, dot_vec, dot_vecs_ls, a_vecs_ls

def test_network_accuracy_mnist(network, actual_table, activation_function):
    num_correct, num_instances, misclassified_ls = 0, len(actual_table), []
    for tup in actual_table:
        x, expected_out = tup
        #print(f"Input: \n{x}")
        #print(f"Expected_out: {expected_out}")
        #print(f"Prediction: {forward_propagate(x, network, activation_function)[0]}")
        if(forward_propagate_mnist(x, network, activation_function)[0] == expected_out): num_correct += 1
        else: misclassified_ls.append(tup)
        #print(f"Current Accuracy: {(num_correct/num_instances)}")
        #input()
    return (num_correct/num_instances), misclassified_ls

def back_propagation_mnist(actual_table, network, num_epochs, lamb, error_threshold, activation_function, activation_function_prime):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    #Training
    network_error, epoch_count = 1, 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        for tup in actual_table:
            x, expected_output = tup
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = forward_propagate_mnist(x, network, activation_function)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            #Backward Propagate - Calculate Gradient Descent Values
            delL_ls = [activation_function_prime(dot_vec)*(expected_output-a_vec)]                        #del_N           #del_Ls[-1] = delN (gradient function for LAST FUNCTION)
            for l in range(len(network)-2, -1, -1):
                print(dot_vecs_ls[l+1])
                print((delL_ls[0] @ (network[l+1][0]).T))
                input()
                delL_vector = activation_function_prime(dot_vecs_ls[l+1]) * (delL_ls[0] @ (network[l+1][0]).T)
                delL_ls = [delL_vector] + delL_ls
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + np.array([[lamb]]) * delL_ls[l]                               #update bias
                layer[0] = layer[0] + np.array([[lamb]]) * ((a_vecs_ls[l]).T @ delL_ls[l])              #update weight
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network_accuracy_mnist(network, actual_table, activation_function)
        network_error = 1 - network_accuracy
        lamb *= 0.99
    return network

def back_propagation_multilayer(actual_table, network, num_epochs, lamb, error_threshold, activation_function, activation_function_prime):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, epoch_count = 1, 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        for tup in actual_table:
            x, expected_output = tup
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = forward_propagate(x, network, activation_function)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            #Backward Propagate - Calculate Gradient Descent Values
            delL_ls = [activation_function_prime(dot_vec)*(expected_output-a_vec)]                          #del_N           #del_Ls[-1] = delN (gradient function for LAST FUNCTION)
            for l in range(len(network)-2, -1, -1):
                print(dot_vecs_ls[l+1])
                print(((network[l+1][0]).T @ delL_ls[0]))
                input()
                delL_vector = activation_function_prime(dot_vecs_ls[l+1]) * ((network[l+1][0]).T @ delL_ls[0])
                delL_ls = [delL_vector] + delL_ls
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + np.array([[lamb]]) * delL_ls[l]                               #update bias
                layer[0] = layer[0] + np.array([[lamb]]) * (delL_ls[l] @ (a_vecs_ls[l]).T)              #update weight
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network_accuracy(network, actual_table, activation_function)
        network_error = 1 - network_accuracy
        lamb *= 0.99
    return network

network2 = create_network([2,2,1])
print_network(network2)

network_mnist = [[layer[0].T, layer[1].T] for layer in network2]
print_network(network_mnist)

xor_actualtable2 = [(np.array([[0],[0]]), 0), (np.array([[0],[1]]), 1), (np.array([[1],[0]]), 1), (np.array([[1],[1]]), 0)]
xor_actualtable_mnist = [(np.array([[0],[0]]).T, 0), (np.array([[0],[1]]).T, 1), (np.array([[1],[0]]).T, 1), (np.array([[1],[1]]).T, 0)]

network_2 = back_propagation_multilayer(xor_actualtable2, network2, 2, 0.1, 0, sigmoid, sigmoid_prime)
network_mnist = back_propagation_mnist(xor_actualtable_mnist, network_mnist, 2, 0.1, 0, sigmoid_mnist, sigmoid_prime_mnist)
print("DONE")
print_network(network_2)
print_network(network_mnist)