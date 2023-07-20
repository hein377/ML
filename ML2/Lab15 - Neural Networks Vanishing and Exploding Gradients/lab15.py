import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np

import neural_network_instantiation
import neural_network_setup
import neural_network_propagation

# QUESTION 1
'''
You will generate 1,000 data points from the two circles problem and rescale
the inputs to the range [-1, 1] before splitting it into train and test sets.
Use half of the data for training and the remaining 500 for the test set.

The model will have an input layer with two inputs, for the two variables in 
the dataset, one hidden layer with five nodes, and an output layer with one 
node used to predict the class probability. The hidden layer will use the 
hyperbolic tangent activation function (tanh) and the output layer will use 
the logistic activation function (sigmoid) to predict class 0 or class 1 or
something in between.

Train your network 500 epochs, and once training is complete, report the
accuracy for both training and test data. Also, line plot of model accuracy
on the train and test sets, showing the change in performance over all 500
training epochs.
'''

# DATASET CREACTION
circle_X, circle_y = make_circles(1000, noise=0.068, factor=0.8)
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(circle_X)
circle_X = scaler.transform(circle_X)

circle_X_train, circle_X_test, circle_y_train, circle_y_test = train_test_split(circle_X, circle_y, test_size=0.5, random_state=1, stratify = circle_y)

# NEURAL NETWORK TRAINING
TRAINING_ACTUAL_TABLE = [(np.array([circle_X_train[i]]).T, np.array([[circle_y_train[i]]])) for i in range(circle_X_train.shape[0])]                         #[ ( input_array <2x1 np.array of ints>, output_array <1x1 np.array of ints> ), ... ]
TESTING_ACTUAL_TABLE = [(np.array([circle_X_test[i]]).T, np.array([[circle_y_test[i]]])) for i in range(circle_X_test.shape[0])]                             #[ ( input_array <2x1 np.array of ints>, output_array <1x1 np.array of ints> ), ... ]
NUM_EPOCHS = 500
LAMB = 5
#MIN_INITIALIZATION_VAL = -1
#MAX_INITIALIZATION_VAL = 1
MIN_INITIALIZATION_VAL = -20
MAX_INITIALIZATION_VAL = 20

activation_functions = [neural_network_setup.tanh, neural_network_setup.sigmoid]
activation_function_primes = [neural_network_setup.tanh_prime, neural_network_setup.sigmoid_prime]

trained_network, training_network_accuracies_ls = neural_network_propagation.back_propagation_multilayer(TRAINING_ACTUAL_TABLE, neural_network_instantiation.create_network([2, 5, 1], MIN_INITIALIZATION_VAL, MAX_INITIALIZATION_VAL), NUM_EPOCHS, LAMB, activation_functions, activation_function_primes, neural_network_setup.test_network_accuracy, verbose=True)

plt.plot([i for i in range(NUM_EPOCHS)], training_network_accuracies_ls)
plt.title("Network Training Accuracies")
plt.xlabel("Num Epochs")
plt.ylabel("Training Accuracy")
plt.show()
print(f"Testing Network Accuracy: {neural_network_setup.test_network_accuracy(trained_network, TESTING_ACTUAL_TABLE, activation_functions)}")