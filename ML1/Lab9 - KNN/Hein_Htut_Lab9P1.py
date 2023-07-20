import numpy as np
import pandas as pd
from tabulate import tabulate
from random import randrange

# FILE SET UP
FILENAME, TRANIINGFILENAME, TESTINGFILENAME = "lab9_iris.csv", "lab9_iris_pytraining.txt", "lab9_iris_pytesting.txt"
IRIS_DF = pd.read_csv(FILENAME)

def pos_values(s): return list(set(s))

def print_pretty_table(dataset, filename):
    f = open(filename, "w")
    f.write(tabulate(dataset, headers='keys', tablefmt='psql'))
    f.close()

#LABELPVALS = IRIS_PVAL[IRIS_DF.columns[-1]]
LABELPVALS = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]
LABEL_NUM, NUM_LABEL = {}, {}
for i in range(len(LABELPVALS)):
    LABEL_NUM.update({LABELPVALS[i]:i})
    NUM_LABEL.update({i:LABELPVALS[i]})
K_NEIGHBORS = 5

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

def stratified_random_sampling(df, prettytablefn, trainingfn, testingfn, testingpercent):
    print_pretty_table(df, prettytablefn)
    pos_label_vals = pos_values(df[df.columns[-1]].to_numpy())
    
    stratified_list = stratify(df, pos_label_vals)
    training_set_df, testing_set_df = split_training_testing(stratified_list, testingpercent)

    training_set_df.to_csv(trainingfn)
    testing_set_df.to_csv(testingfn)
    #print_pretty_table(training_set_df, trainingfn)
    #print_pretty_table(testing_set_df, testingfn)
    
    return training_set_df, testing_set_df

#training_df, testing_df = stratified_random_sampling(IRIS_DF, "lab8_iris_prettytable.txt", TRANIINGFILENAME, TESTINGFILENAME, 0.33)
training_df, testing_df = pd.read_csv(TRANIINGFILENAME), pd.read_csv(TESTINGFILENAME)
training_df, testing_df = training_df.drop(training_df.columns[0], axis = 1), testing_df.drop(testing_df.columns[0], axis = 1)

# KNN CLASSIFIER
from scipy.spatial.distance import euclidean
def add_dic(dic, key):
    if(key in dic): dic[key] += 1
    else: dic.update({key:1})

def knn_choose_best(sorted_ls):         #sorted_ls = [ instance1<tup>, instance2<tup>, ... , instancek<tup> ] sorted by smallest to largest distance
    print(sorted_ls)
    labelval_freq = {}
    for instance in sorted_ls: add_dic(labelval_freq, instance[-1])
    print(labelval_freq)
    input()
    return sorted(labelval_freq, key = lambda x: labelval_freq[x], reverse = True)[0]           #return class label value with the highest frequency

def knn_classify(training_df, testing_instance, k_num):                  #testing_instance <tup>
    traininginstance_distance = {}
    for index, training_instance in training_df.iterrows():
        training_instance = tuple(training_instance)
        if(training_instance not in traininginstance_distance): traininginstance_distance.update({training_instance:euclidean(training_instance[:-1], testing_instance[:-1])})                #instance[-1] to ensure class label value is not used to calculate euclidean distance

    print(sorted(traininginstance_distance, key = lambda x: traininginstance_distance[x])[:10])
    return knn_choose_best(sorted(traininginstance_distance, key = lambda x: traininginstance_distance[x])[:k_num])         #only gives K instances that are sorted by shortest distance

def knn_test(training_df, testing_df, k_num):
    cmatrix = np.zeros(((n:=len(LABEL_NUM.keys())), n))               #n=num_instances
    for index, row in testing_df.iterrows():
        model_prediction, true_label_val = knn_classify(training_df, tuple(row), k_num), row[-1]
        x, y = LABEL_NUM[true_label_val], LABEL_NUM[model_prediction]
        cmatrix[x,y] += 1
    return cmatrix, (np.trace(cmatrix)/np.sum(cmatrix))             # accrate = sum(diagonal)/sum(matrix elements)

def calcMacrosMicros(cmatrix):
    precision, recall = [], []
    coltotals, rowtotals, diagonalvals = cmatrix.sum(axis=0).tolist(), cmatrix.sum(axis=1).tolist(), cmatrix.diagonal().tolist()
    for i in range(len(diagonalvals)):
        precision.append((diagonal_val:=diagonalvals[i])/rowtotals[i])
        recall.append((diagonal_val/coltotals[i]))
    
    return sum(precision)/len(precision), sum(recall)/len(recall), sum(diagonalvals)/sum(rowtotals), sum(diagonalvals)/sum(coltotals)           #returns macprec, macrecall, micprec, micrecall

def knn(training_df, testing_df, k_num):
    cmatrix, testing_accRate = knn_test(training_df, testing_df, k_num)
    macroPrecision, macroRecall, microPrecision, microRecall = calcMacrosMicros(cmatrix)

    print("KNN", end="\n")
    print(f"Testing Accuracy: {testing_accRate}")
    row_labels = col_labels = list(LABEL_NUM.keys())
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(pd.DataFrame(cmatrix, columns=col_labels, index=row_labels))
    print(f"Macroaverage Precision: {macroPrecision},   Macroaverage Recall: {macroRecall}")
    print(f"Microaverage Precision: {microPrecision},   Microaverage Recall: {microRecall}")

#knn(training_df, testing_df, K_NEIGHBORS)

# TEST OUT DiFF K VALS
import matplotlib.pyplot as plt
def testknn(training_df, testing_df, k_num):
    cmatrix, testing_accRate = knn_test(training_df, testing_df, k_num)
    return testing_accRate

def testK_knn(training_df, testing_df, kvalLB, kvalUB):
    x, y = [i for i in range(kvalLB, kvalUB)], []
    for kval in x: y.append(testknn(training_df, testing_df, kval))
    plt.scatter(x, y, s=5, alpha=0.5)
    plt.show()

testK_knn(training_df, testing_df, 1, 50)