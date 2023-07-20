import numpy as np
import math
import random

# NEURAL NETWORKS CREATION
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

# NEURAL NETWORKS SETUP
def step(dot):
    return np.array([[1 if dot[i][0] >= 0 else 0 for i in range(dot.shape[0])]]).T

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

def choose_best(network_output):
    network_output = network_output[0][0]
    if(network_output>=0.5): return 1
    return 0

def test_network_accuracy_smoothactivationfunction(network, actual_table, activation_function):
    num_correct, num_instances, misclassified_ls = 0, len(actual_table), []
    for tup in actual_table:
        x, expected_out = tup
        #print(f"Input: \n{x}")
        #print(f"Expected_out: {expected_out}")
        #print(f"Prediction: {forward_propagate(x, network, activation_function)[0]}")
        if(choose_best(forward_propagate(x, network, activation_function)[0]) == expected_out): num_correct += 1
        else: misclassified_ls.append(tup)
        #print(f"Current Accuracy: {(num_correct/num_instances)}")
        #input()
    return (num_correct/num_instances), misclassified_ls

# NEURAL NETWORKS
def back_propagation_testing_singlelayer(actual_table, network, num_epochs, lamb, error_threshold, activation_function):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, epoch_count = 1, 0
    print_network(network)
    input()
    while(epoch_count < num_epochs and network_error > error_threshold):
        print(f"EPOCH: {epoch_count}")
        print(f"Current Network Error: {network_error}")
        for tup in actual_table:
            x, cur_expected_output = tup
            print(f"Expected_output: {cur_expected_output}")
            print(f"Input:")
            print(x)
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = forward_propagate(x, network, activation_function)                        #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            target_error = cur_expected_output-a_vec
            print(f"Network_output: {a_vec}")
            print(f"Error: {target_error}", end="\n\n")
            #Backward Propagate - Update Values
            for l in range(len(network)):
                print(f"Layer{l} Update")
                layer = network[l]
                layer[1] = layer[1] + lamb * (target_error)                          #update bias
                layer[0] = layer[0] + lamb * (target_error) * x.T                  #update weight
                print(f"W_n: {layer[0]}")
                print(f"b_n: {layer[1]}")
            input()
        print("----------------------")
        epoch_count += 1
        #print("Testing network:")
        network_accuracy, misclassified_ls = test_network_accuracy(network, actual_table, activation_function)
        network_error = 1 - network_accuracy
        lamb *= 0.99
        print(f"Current Network Error: {network_error}")
        print()
    return network

def back_propagation_singlelayer(actual_table, network, num_epochs, lamb, error_threshold, activation_function):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, epoch_count = 1, 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        for tup in actual_table:
            x, expected_output = tup
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = forward_propagate(x, network, activation_function)                        #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            target_error = expected_output-a_vec
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + lamb * (target_error)                          #update bias
                layer[0] = layer[0] + lamb * (target_error) * x.T                  #update weight
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network_accuracy(network, actual_table, activation_function)
        network_error = 1 - network_accuracy
    return network

def back_propagation_testing_multilayer(actual_tables, network, num_epochs, lamb, error_threshold, activation_function):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, epoch_count = 1, 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        print(f"EPOCH: {epoch_count}")
        print(f"Current Network Error: {network_error}")
        for tup in actual_tables[-1][0]:
            x, cur_expected_output = tup
            print(f"Input:")
            print(x)
            print()
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = forward_propagate(x, network, activation_function)                        #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            #Backward Propagate - Update Values 
            for l in range(len(network)):
                print(f"LAYER{l} UPDATE")
                layer, layer_actual_tables = network[l], actual_tables[l]
                layer_expected_outputs = np.empty(shape = [len(layer_actual_tables),1]) 
                for i in range(len(layer_actual_tables)):
                    perceptron_table = layer_actual_tables[i]
                    x_index = binary_to_decimal(''.join([str(x[row][0]) for row in range(len(x))]))
                    x, expected_output = perceptron_table[x_index]
                    layer_expected_outputs[i][0] = expected_output
                layer_target_errors = layer_expected_outputs-a_vecs_ls[l+1]

                print("Layer output:")
                print(a_vecs_ls[l+1])
                print("Layer Expected_output: ")
                print(layer_expected_outputs)
                print("Layer Error: ")
                print(layer_target_errors)

                print(f"W_cur: {layer[0]}")
                print(f"b_cur: {layer[1]}")
                layer[1] = layer[1] + np.array([[lamb]]) * layer_target_errors                               #update bias
                layer[0] = layer[0] + np.array([[lamb]]) * ((a_vecs_ls[l+1]).T @ layer_target_errors)              #update weight
                print(f"W_n: {layer[0]}")
                print(f"b_n: {layer[1]}")
                print()
            input()
        print("----------------------")
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network_accuracy(network, actual_tables[-1][0], activation_function)
        network_error = 1 - network_accuracy
        lamb *= 0.99
        print(f"Current Network Error: {network_error}")
        print()
    return network

'''def back_propagation_multilayer(actual_tables, network, num_epochs, lamb, error_threshold, activation_function):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, epoch_count = 1, 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        for tup in actual_tables[-1][0]:
            x, cur_expected_output = tup
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = forward_propagate(x, network, activation_function)                        #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            #Backward Propagate - Update Values 
            for l in range(len(network)):
                layer, layer_actual_tables = network[l], actual_tables[l]
                layer_expected_outputs = np.empty(shape = [len(layer_actual_tables),1]) 
                for i in range(len(layer_actual_tables)):
                    perceptron_table = layer_actual_tables[i]
                    x_index = binary_to_decimal(''.join([str(x[row][0]) for row in range(len(x))]))
                    x, expected_output = perceptron_table[x_index]
                    layer_expected_outputs[i][0] = expected_output
                layer_target_errors = layer_expected_outputs-a_vecs_ls[l+1]

                layer[1] = layer[1] + np.array([[lamb]]) * layer_target_errors                               #update bias
                layer[0] = layer[0] + np.array([[lamb]]) * ((a_vecs_ls[l+1]).T @ layer_target_errors)              #update weight
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network_accuracy(network, actual_tables[-1][0], activation_function)
        network_error = 1 - network_accuracy
        lamb *= 0.99
    return network'''

def back_propagation_multilayer(actual_table, network, num_epochs, lamb, error_threshold, activation_function, activation_function_prime, test_network):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, misclassified_ls = test_network(network, actual_table, activation_function)
    epoch_count = 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        for tup in actual_table:
            x, expected_output = tup
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = forward_propagate(x, network, activation_function)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            #Backward Propagate - Calculate Gradient Descent Values
            delL_ls = [activation_function_prime(dot_vec)*(expected_output-a_vec)]                          #del_N           #del_Ls[-1] = delN (gradient function for LAST FUNCTION)
            for l in range(len(network)-2, -1, -1):
                delL_vector = activation_function_prime(dot_vecs_ls[l+1]) * ((network[l+1][0]).T @ delL_ls[0])
                delL_ls = [delL_vector] + delL_ls
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + np.array([[lamb]]) * delL_ls[l]                               #update bias
                layer[0] = layer[0] + np.array([[lamb]]) * (delL_ls[l] @ (a_vecs_ls[l]).T)              #update weight
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network(network, actual_table, activation_function)
        network_error = 1 - network_accuracy
        lamb *= 0.999
    return network

# QUESTION 1 PART A
num_epoch, lamb, error_threshold = 50, 0.1, 0

#OR Function
or_actualtable = [(np.array([[0],[0]]), 0), (np.array([[0],[1]]), 1), (np.array([[1],[0]]), 1), (np.array([[1],[1]]), 1)]
or_trained_network = back_propagation_singlelayer(or_actualtable, create_network([2,1]), num_epoch, lamb, error_threshold, step)

#AND Function
and_actualtable = [(np.array([[0],[0]]), 0), (np.array([[0],[1]]), 0), (np.array([[1],[0]]), 0), (np.array([[1],[1]]), 1)]
and_trained_network = back_propagation_singlelayer(and_actualtable, create_network([2,1]), num_epoch, lamb, error_threshold, step)

#NAND Function
nand_actualtable = [(np.array([[0],[0]]), 1), (np.array([[0],[1]]), 1), (np.array([[1],[0]]), 1), (np.array([[1],[1]]), 0)]
nand_trained_network = back_propagation_singlelayer(nand_actualtable, create_network([2,1]), num_epoch, lamb, error_threshold, step)

def test_binary_tables(network, actual_table, activation_function):
    num_correct, num_instances, misclassified_ls = 0, len(actual_table), []
    for tup in actual_table:
        x, expected_out = tup
        if((prediction:=forward_propagate(x, network, activation_function)[0]) == expected_out): num_correct += 1
        else: misclassified_ls.append(tup)
        print("Input: {}^T     Actual: {}   Prediction:{} ".format(x.T, expected_out, prediction))
    return (num_correct/num_instances), misclassified_ls

print("OR")
or_network_accuracy, or_misclassified_ls = test_binary_tables(or_trained_network, or_actualtable, step)
print("\nAND")
and_network_accuracy, and_misclassified_ls = test_binary_tables(and_trained_network, and_actualtable, step)
print("\nNAND")
nand_network_accuracy, nand_misclassified_ls = test_binary_tables(nand_trained_network, nand_actualtable, step)

# QUESTION 1 PART B
# GENERATE TRUTH TABLE
def binary_to_decimal(binary):                  #binary = string, decimal = int
    binaryls, decimal = list(binary), 0
    for i in range(length:=len(binaryls)):
        val = int(binaryls[i])
        decimal += val * (2**(length-1-i))
    return decimal

def decimal_to_binary(decimal, num_bits):       #binary = string, decimal = int
    binary = ""
    for i in range(num_bits-1, -1, -1):
        if(decimal - 2**i >= 0): 
            binary += "1"
            decimal -= 2**i
        else: binary += "0"
    return binary

def create_truth_table(num_bits, canonical_int):            #returns [ ( np.array([[input]]).T , output ) <tuples of nx1 input_vector, output_value> ]; e.g. if num_bits=3: [ (np.array([[1, 1, 1]]).T, 0), (np.array([[1, 1, 0]]).T, 1) ... ] from largest to smallest
    ind, binary_int = 2**num_bits - 1, list(decimal_to_binary(canonical_int, 2**num_bits))
    return [(np.array([[int(n) for n in list(decimal_to_binary(i, num_bits))]]).T,int(binary_int[i])) for i in range(ind, -1, -1)]

def pretty_print_tt(table):
    for inputs, output in table:
        for i in inputs: print(i, end = "  ")
        print(f"|  {output}")

three_bit_and_trained_network = back_propagation_singlelayer((three_bit_and_actualtable:=create_truth_table(3, 8)), create_network([3,1]), num_epoch, lamb, error_threshold, step)
print("\nAND")
three_bit_and_network_accuracy, three_bit_and_misclassified_ls = test_binary_tables(three_bit_and_trained_network, three_bit_and_actualtable, step)

# QUESTION 2 PART A
#XOR Function
def generate_truth_table_single_perceptron(network, activation_function):
    truth_table, inputs = [], [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]), np.array([[1],[1]])]
    for x in inputs: truth_table.append((x, forward_propagate(x, network, activation_function)[0]))
    return truth_table

def test_binary_tables_smoothactivationfunction(network, actual_table, activation_function):
    num_correct, num_instances, misclassified_ls = 0, len(actual_table), []
    for tup in actual_table:
        x, expected_out = tup
        if((prediction:=choose_best(forward_propagate(x, network, activation_function)[0])) == expected_out): num_correct += 1
        else: misclassified_ls.append(tup)
        print(f"Raw Network Output: {forward_propagate(x, network, activation_function)[0]}")
        print("Input: {}^T     Actual: {}   Prediction:{} ".format(x.T, expected_out, prediction))
    return (num_correct/num_instances), misclassified_ls

#XOR Function
lamb, error_threshold = 15, 0
#xor_actualtables = [ [ generate_truth_table_single_perceptron([[np.array([[1,1]]),-0.5]], step), generate_truth_table_single_perceptron([[np.array([[-1,-1]]),1.5]], step) ] , [[(np.array([[0],[0]]), 0), (np.array([[0],[1]]), 1), (np.array([[1],[0]]), 1), (np.array([[1],[1]]), 0)]] ]           # actual_tables = [ layer_1_actualtables, layer_2_actualtables, ... output_layer_actualtables ] where actualtables = [ perceptron1_actualtables, perceptron2_actualtables, ... ]
xor_actualtable = [(np.array([[0],[0]]), 0), (np.array([[0],[1]]), 1), (np.array([[1],[0]]), 1), (np.array([[1],[1]]), 0)]
xor_trained_network = back_propagation_multilayer(xor_actualtable, create_network([2,2,1]), 1000, 7.5, error_threshold, sigmoid, sigmoid_prime, test_network_accuracy_smoothactivationfunction)
print("\nXOR")
xor_network_accuracy, xor_misclassified_ls = test_binary_tables_smoothactivationfunction(xor_trained_network, xor_actualtable, sigmoid)

#XNOR Function
#xnor_actualtables = [ [ generate_truth_table_single_perceptron([[np.array([[-1,-1]]),0.5]], step), generate_truth_table_single_perceptron([[np.array([[1,1]]),-1.5]], step) ] , [[(np.array([[0],[0]]), 1), (np.array([[0],[1]]), 0), (np.array([[1],[0]]), 0), (np.array([[1],[1]]), 1)]] ]           # actual_tables = [ layer_1_actualtables, layer_2_actualtables, ... output_layer_actualtables ] where actualtables = [ perceptron1_actualtables, perceptron2_actualtables, ... ]
xnor_actualtable = [(np.array([[0],[0]]), 1), (np.array([[0],[1]]), 0), (np.array([[1],[0]]), 0), (np.array([[1],[1]]), 1)]
xnor_trained_network = back_propagation_multilayer(xnor_actualtable, create_network([2,2,1]), 1000, 7.5, error_threshold, sigmoid, sigmoid_prime, test_network_accuracy_smoothactivationfunction)
print("\nXNOR")
xnor_network_accuracy, xnor_misclassified_ls = test_binary_tables_smoothactivationfunction(xnor_trained_network, xnor_actualtable, sigmoid)

four_bit_xnor_trained_network = back_propagation_multilayer((four_bit_xnor_actualtable:=create_truth_table(4, 38505)), create_network([4,1]), 300, 10, 0, sigmoid, sigmoid_prime, test_network_accuracy_smoothactivationfunction)
print("\nXNOR")
pretty_print_tt(four_bit_xnor_actualtable)

#QUESTION 3 PART C
#DISJUNCTIVE NORMAL FORM MLP
w1, b1 = np.array([[-1,-1,1,1,-1], [-1,1,-1,1,1], [-1,1,1,-1,-1], [1,-1,-1,-1,1], [1,-1,1,1,1], [1,1,-1,-1,1]]), np.array([[-2], [-3], [-2], [-2], [-4], [-3]])
w2, b2 = np.array([[1,1,1,1,1,1]]), np.array([[-1]])
layer1 = [w1,b1]
layer2 = [w2,b2]
network = layer1, layer2
dnf_actual_table = create_truth_table(5, 35144000)
dnf_trained_newtork = back_propagation_multilayer(dnf_actual_table, network, 50, 0.1, 0, step, step, test_network_accuracy)
print("\nDISJUNCTIVE NORMAL FORM TRUTH TABLE")
test_binary_tables(dnf_trained_newtork, dnf_actual_table, step)