'''
DEMO
# split a dataset into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# create dataset
x, y = make_blobs(n_samples=1000)                                                    #x is 2x2 list (matrix) and y is a normal list where x is training data and y is class label data
# split into train test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.67)          #same effect as line 7
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
'''
from cProfile import label
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import pandas as pd 
import numpy as np
from random import randrange

def print_pretty_table(dataset, filename):
    f = open(filename, "w")
    f.write(tabulate(dataset, headers='keys', tablefmt='psql'))
    f.close()

def pos_label_values(df): return list(set(df[df.columns[-1]].to_numpy()))

# SIMPLE RANDOM SAMPLING
def simple_random_sampling(dataset_filename, prettytable_filename, training_filename, testing_filename, testing_percent):
    dataset = pd.read_csv(dataset_filename)
    print_pretty_table(dataset, prettytable_filename)
    attribute_data = dataset.iloc[:,[i for i in range(len(dataset.columns)-1)]].to_numpy()                   #all attributes except class label column (0 to n-1 columns)
    label_data = dataset[dataset.columns[-1]].to_numpy()                                                     #only class label column (n column)

    x_train, x_test, y_train, y_test = train_test_split(attribute_data, label_data, test_size=testing_percent)
    training_set = np.c_[x_train, y_train]          #training_set <2x2 lists>; combines attribute data columns for training and class label data column for testing
    testing_set = np.c_[x_test, y_test]             #testing_set <2x2 lists>; combines attribute data columns for testing and class label data column for testing
    training_set_df = pd.DataFrame(training_set, columns=dataset.columns)
    testing_set_df = pd.DataFrame(testing_set, columns=dataset.columns)

    print_pretty_table(training_set_df, training_filename)
    print_pretty_table(testing_set_df, testing_filename)

simple_random_sampling("lab5_iris.csv", "lab5_iris_prettytable.txt", "lab5_iris_pytraining.txt", "lab5_iris_pytesting.txt", 0.33)

# STRATIFIED RANDOM SAMPLING
def stratify(df, pos_label_vals):                   #df<pandas dataframe obj> = dataframe, pos_label_values<list>; returns [df1<dataframe>, df2<dataframe>, ...] where dfs are separated by class label
    stratified_list = []
    for pos_val in pos_label_vals: stratified_list.append(df.loc[df[df.columns[-1]] == pos_val])
    return stratified_list

def random_indices(indices, n):
    training_set_indices, testing_set_indices = indices, []
    for i in range(n): testing_set_indices.append(training_set_indices.pop(randrange(len(training_set_indices))))
    return training_set_indices, testing_set_indices

def split_training_testing(stratified_ls, testingpercent):
    training_set_ls, testing_set_ls = [], []
    for label_df in stratified_ls:
        n = int(testingpercent * len(label_df))
        training_set_indices, testing_set_indices = random_indices(list(range(len(label_df))), n)
        training_set_ls.append(label_df.iloc[training_set_indices,:])
        testing_set_ls.append(label_df.iloc[testing_set_indices,:])
    return pd.concat(training_set_ls), pd.concat(testing_set_ls)

def stratified_random_sampling(datafn, prettytablefn, trainingfn, testingfn, testingpercent):
    df = pd.read_csv(datafn)
    print_pretty_table(df, prettytablefn)
    pos_label_vals = pos_label_values(df)
    
    stratified_list = stratify(df, pos_label_vals)
    training_set_df, testing_set_df = split_training_testing(stratified_list, testingpercent)

    print_pretty_table(training_set_df, trainingfn)
    print_pretty_table(testing_set_df, testingfn)

stratified_random_sampling("lab5_unbalanced.csv", "lab5_unbalanced_prettytable.txt", "lab5_unbalanced_pytraining.txt", "lab5_unbalanced_pytesting.txt", 0.33)