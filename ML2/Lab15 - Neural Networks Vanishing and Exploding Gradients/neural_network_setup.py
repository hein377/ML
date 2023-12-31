import numpy as np
import math

# NEURAL NETWORKS SETUP
def step(dot):
    return np.array([[1 if dot[i][0] >= 0 else 0 for i in range(dot.shape[0])]]).T

def sigmoid(dot):                       #input: np.array(); returns a np.array() w/ sigmoid function applied to each entry
    arr = [(1/(1+math.exp(-element))) for element in dot]
    return np.array([arr]).T

def sigmoid_prime(dot):
    return (sig:=sigmoid(dot)) * (1-sig)

def tanh(dot):                          #input: np
    return np.tanh(dot)

def tanh_prime(dot):
    return (np.ones(dot.shape)-tanh(dot)**2)

def relu(dot):
    return np.maximum(0, dot)

def relu_prime(dot):
    return np.array([[0 if element<=0 else 1 for element in dot]]).T

def find_dot(w, b, x):                            #@ is normal matrix multiplication
    return (w@x) + b

def perceptron(A, dot): return A(dot)           #returns f* = A(w dot x + b)<np.array>

def forward_propagate(x, network, activation_functions):                      #propagates through network and returns dot_vecs_ls and a_vecs_ls of; values of each layer
    a_vec = x
    dot_vecs_ls, a_vecs_ls = [np.array([[0, 0]])], [a_vec]         #don't use dot_vec[0]; just a placeholder
    for i in range(len(network)):
        layer = network[i]
        w_matrix, b_scalar = layer[0], layer[1]
        dot_vec = find_dot(w_matrix, b_scalar, a_vec)
        a_vec = perceptron(activation_functions[i], dot_vec)
        dot_vecs_ls.append(dot_vec)
        a_vecs_ls.append(a_vec)
    return a_vec, dot_vec, dot_vecs_ls, a_vecs_ls

def choose_best(ls):                    #returns index with highest value in ls
    return ls.index(max(ls))

def calc_error(out, expected_out, length_of_arrays):                          #input: np.arrays(), both same size of length_of_arrays
    return 1/length_of_arrays * (np.linalg.norm(expected_out - out))**2

def make_highest_one(arr):                #returns same size numpy array with 0's everywhere except 1 at the index with the max value
    max_val_index = arr.T[0].tolist().index(max(arr))
    return np.array([[1] if x==max_val_index else [0] for x in range(len(arr))])

def make_final_prediction(x):
    if x < 0.5: return 0
    else: return 1

def test_network_accuracy(network, actual_table, activation_functions):
    num_correct, num_instances = 0, len(actual_table)
    for tup in actual_table:
        x, expected_out = tup
        if(make_final_prediction(forward_propagate(x, network, activation_functions)[0]) == expected_out): num_correct += 1
    return (num_correct/num_instances)