import numpy as np
import matplotlib.pyplot as plt

import pickle

import neural_network_instantiation
import neural_network_setup
import neural_network_propagation

#MNIST DATASET PROCESSING
def one_at_index(ind):
    ls = [0] * 10
    ls[int(ind)] = 1
    return ls

def div(ls, val): return [int(ls[i]) / val for i in range(len(ls))]

def process_data(filename):
    ls = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split(",")
            inp, output = np.array([div(line[1:], 255)]).T, np.array([one_at_index(line[0])]).T                 #inp = input (input is a Python keyword and messes stuff up so use inp instead)
            ls.append((inp, output))
    return ls

#VARIABLES
ACTUAL_TABLE = process_data("mnist_train.csv")                          #[ ( input_array <784x1 np.array of ints>, output_array <10x1 np.array of ints> ), ... ]
NUM_EPOCHS = 65
NUM_WEIGHTS = 5

#NEURAL NETWORK TRAINING
neural_network_propagation.back_propagation_multilayer_MNIST(ACTUAL_TABLE, NUM_EPOCHS, NUM_WEIGHTS, neural_network_setup.sigmoid, neural_network_setup.sigmoid_prime, neural_network_setup.test_network_accuracy_MNIST, neural_network_instantiation.create_network([784, 300, 100, 10]), [], [], [], [[] for i in range(NUM_WEIGHTS)], [[] for i in range(NUM_WEIGHTS)], [0.1])

'''savefile = open('network2.pkl', 'rb')
network, network_accuracies, network_errorsTotal, network_errorsTotal_sum, w_h1i_vals, w_h1i_gradient_vals, lambda_vals = pickle.load(savefile)
#neural_network_propagation.back_propagation_multilayer_MNIST(ACTUAL_TABLE, NUM_EPOCHS, neural_network_setup.sigmoid, neural_network_setup.sigmoid_prime, neural_network_setup.test_network_accuracy_MNIST, network, network_accuracies, network_errorsTotal, network_errorsTotal_sum, w_h1i_vals, w_h1i_gradient_vals, lambda_vals)

#NEURAL NETWORK TESTING
testing_accuracy = neural_network_setup.test_network_accuracy_MNIST(network, ACTUAL_TABLE, neural_network_setup.sigmoid)
print(f"TESTING ACCURACY: {testing_accuracy}")

#GRAPHS
def plot_line_graph(x_list, y_list, x_name, y_name):
    plt.plot(x_list, y_list, color='blue', marker='o')
    plt.title(f"{y_name} vs {x_name}")
    plt.xlabel(f"{x_name}")
    plt.ylabel(f"{y_name}")
    plt.grid(True)
    plt.show()

epochs = [i for i in range(len(network_accuracies))]
plot_line_graph(epochs, network_accuracies, "Epoch", "Network_Accuracy")
plot_line_graph(epochs, network_errorsTotal_sum, "Epoch", "Network_Error_Total_Sum")
for i in range(NUM_WEIGHTS): plot_line_graph(w_h1i_vals[i], network_errorsTotal, f"W_h1,i{i}", "Network_Error_Total")

print("Wh1_i Vals")
for i in range(NUM_WEIGHTS): 
    print(f"Wh1_i{i} Minimum: {min(w_h1i_vals[i])}")
    print(f"Wh1_i{i} Maximum: {max(w_h1i_vals[i])}")
    print()

print("Wh1_i Gradient Vals")
for i in range(NUM_WEIGHTS): 
    print(f"dError/dWh1_i{i} Minimum: {min(w_h1i_gradient_vals[i])}")
    print(f"dError/dWh1_i{i} Maximum: {max(w_h1i_gradient_vals[i])}")
    print()'''