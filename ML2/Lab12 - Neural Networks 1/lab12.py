import numpy as np
import math
import random

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# Dataset Creation
X, y = make_blobs(n_samples=300, n_features=2, cluster_std=0.75
                  ,centers= [(1,3.5), (2,0.5)])
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired);

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
training_df = pd.DataFrame(X_train)
training_df['2'] = y_train
testing_df = pd.DataFrame(X_test)
testing_df['2'] = y_test

# Make Actual Table
def make_actual_table(df):
    actual_table = []
    for ind, row in df.iterrows():
        row = row.tolist()
        actual_table.append((np.array([row[:-1]]), int(row[-1])))
    return actual_table

training_actual_table = make_actual_table(training_df)
testing_actual_table = make_actual_table(testing_df)
print(training_actual_table)

def step(num):
    if(num > 0): return 1
    return 0

def sigmoid(num):                       #input: np.array(); returns a np.array() w/ sigmoid function applied to each entry
    x = (1/(1+math.exp(-num)))
    if(x > 0.5): return 1
    return 0

def find_dot(w, b, x): return (w@x) + b

def perceptron(A, dot): return A(dot)           #returns f* = A(w dot x + b)<np.array>

def forward_propagate(w, b, x): 
    #return perceptron(step, find_dot(w, b, x))
    return perceptron(sigmoid, find_dot(w, b, x))

def choose_best(ls):                    #returns index with highest value in ls
    return ls.index(max(ls))

def test_network_accuracy(network, actual_table):
    num_correct, num_instances, misclassified_ls = 0, len(actual_table), []
    for tup in actual_table:
        x, expected_out = tup
        x = x.T
        if(forward_propagate(network[0][0], network[0][1], x) == expected_out): num_correct += 1
        else: misclassified_ls.append(tup)
    return (num_correct/num_instances), misclassified_ls

def back_propagation_testing(actual_table, network, lamb, error_threshold):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, epoch_count = 1, 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        print("Epoch: {}".format(epoch_count))
        print("Current Network Error: {}".format(network_error), end = "\n\n")
        for tup in actual_table:
            x, expected_output = tup
            x = x.T
            print("p: {}     t: {}".format(x, expected_output))
            #Forward Propagate
            a_vec = forward_propagate(network[0][0], network[0][1], x)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            target_error = expected_output-a_vec
            print("a: {}".format(a_vec))
            print("e: {}".format(target_error))
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + lamb * (target_error)                          #update bias
                layer[0] = layer[0] + lamb * (target_error) * x.T                  #update weight
                print("W_n: {}".format(layer[0]))
                print("b_n: {}".format(layer[1]), end = "\n\n")
            input()
        print("----------------------")
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network_accuracy(network, actual_table)
        network_error = 1 - network_accuracy
        print("Current Network Error: {}".format(network_error))
    return network, misclassified_ls

def back_propagation(actual_table, network, lamb, error_threshold):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, epoch_count = 1, 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        for tup in actual_table:
            x, expected_output = tup
            x = x.T
            #Forward Propagate
            a_vec = forward_propagate(network[0][0], network[0][1], x)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            target_error = expected_output-a_vec
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + lamb * (target_error)                          #update bias
                layer[0] = layer[0] + lamb * (target_error) * x.T                  #update weight
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network_accuracy(network, actual_table)
        network_error = 1 - network_accuracy
    return network, misclassified_ls

num_epochs, learning_rate, error_threshold = 1, 15, 0.05
weight_vector, bias_scalar = np.array([[0, 0]]), np.array([0])     #weight_vector is 1x2 row vector
network = [[weight_vector, bias_scalar]]
trained_network, training_misclassified_ls = back_propagation(training_actual_table, network, learning_rate, error_threshold)

'''points = [(np.array([i])@(trained_network[0][0].T) + trained_network[0][1])[0][0] for i in X]
print(perceptron_equation)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired);
plt.plot(X, perceptron_equation, '-r', label='Decision Boundary')'''

print(test_network_accuracy(trained_network, testing_actual_table), end = "\n\n")

# IRIS DATASET
# Iris Dataset
iris_df = pd.read_csv("iris.csv")
iris_df[iris_df.columns[-1]] = iris_df[iris_df.columns[-1]].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0,1,2])
X, y = iris_df.iloc[:,:-1].to_numpy(), iris_df.iloc[:,-1:].to_numpy()
NUM_DISTINCT_CLASS_VALS = 3

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
training_df = pd.DataFrame(X_train)
training_df['2'] = y_train
testing_df = pd.DataFrame(X_test)
testing_df['2'] = y_test

def one_at_index(ind):
    ls = [0] * NUM_DISTINCT_CLASS_VALS
    ls[int(ind)] = 1
    return ls

# Make Actual Table
def make_actual_table(df):
    actual_table, index_ls = [], [i for i in range(len(df.index))]
    random.shuffle(index_ls)
    for ind in index_ls:
        row = df.iloc[[ind]].values.tolist()[0]
        actual_table.append((np.array([row[:-1]]), np.array([one_at_index(int(row[-1]))]).T))
    return actual_table

training_actual_table = make_actual_table(training_df)
testing_actual_table = make_actual_table(testing_df)
print(training_actual_table)

def sigmoid(dot):                       #input: np.array(); returns a np.array() w/ sigmoid function applied to each entry
    arr = [(1/(1+math.exp(-dot[i][0]))) for i in range(dot.shape[0])]
    return np.array([arr]).T

def find_dot(w, b, x): return (w@x) + b

def perceptron(A, dot): return A(dot)           #returns f* = A(w dot x + b)<np.array>

def forward_propagate(w, b, x): return perceptron(sigmoid, find_dot(w, b, x))

def choose_best(ls):                    #returns index with highest value in ls
    return ls.index(max(ls))

def make_highest_one(ls):                #finds ind w/ max value in the list at returns a list of 0's with 1 at that index
    ls = ls.T[0].tolist()
    temp_ls = [0] * len(ls)
    temp_ls[ls.index(max(ls))] = 1
    return temp_ls

def test_network_accuracy(network, actual_table):
    num_correct, num_instances, misclassified_ls = 0, len(actual_table), []
    for tup in actual_table:
        x, expected_out = tup
        x, expected_out = x.T, expected_out.T[0].tolist()
        prediction = make_highest_one(forward_propagate(network[0][0], network[0][1], x))
        if(prediction == expected_out): num_correct += 1
        else: misclassified_ls.append(tup)
    return (num_correct/num_instances), misclassified_ls

def back_propagation(actual_table, network, lamb, error_threshold):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, epoch_count = 1, 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        for tup in actual_table:
            x, expected_output = tup
            x = x.T
            #Forward Propagate
            a_vec = forward_propagate(network[0][0], network[0][1], x)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            target_error = expected_output-a_vec
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + lamb * (target_error)                          #update bias
                layer[0] = layer[0] + lamb * (target_error) * x.T                  #update weight
        epoch_count += 1
        network_accuracy, misclassified_ls = test_network_accuracy(network, actual_table)
        network_error = 1 - network_accuracy
    return network, misclassified_ls

def create_network(layer_sizes):        #layer_sizes = [input_layer_size, layer1_size, layer2_size, ... , layern_size]
    network = []
    for i in range(1, len(layer_sizes)):
        size1 = layer_sizes[i-1] 
        size2 = layer_sizes[i]                      #current layer size
        wm, bs = create_random_wm(size1, size2), create_random_bs(size2)
        network.append([wm, bs])
    return network    #size1, size2 = 4, 3

def create_random_wm(layer1_size, layer2_size):
    ls = []
    for row in range(layer2_size): ls.append([random.uniform(-1, 1) for i in range(layer1_size)])
    return np.array(ls)

def create_random_bs(layer_size): return np.array([[random.uniform(-1, 1) for i in range(layer_size)]]).T

num_epochs, learning_rate, error_threshold = 100, 0.5, 0.01
network = create_network([4, 3])
trained_network, training_misclassified_ls = back_propagation(training_actual_table, network, learning_rate, error_threshold)
print(test_network_accuracy(trained_network, testing_actual_table), end = "\n\n")