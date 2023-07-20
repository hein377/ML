import pandas as pd
from tabulate import tabulate
from random import randrange

# FILE SET UP
FILENAME, TRANIINGFILENAME, TESTINGFILENAME = "lab9_iris.csv", "lab9_iris_pytraining.txt", "lab9_iris_pytesting.txt"
IRIS_DF = pd.read_csv(FILENAME)

def pos_values(s): 
    pvals = list(set(s))
    pvals.sort()
    return pvals

def print_pretty_table(dataset, filename):
    f = open(filename, "w")
    f.write(tabulate(dataset, headers='keys', tablefmt='psql'))
    f.close()

# DATA PREPROCESSING
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

#training_df, testing_df = stratified_random_sampling(IRIS_DF, "lab9_iris_prettytable.txt", TRANIINGFILENAME, TESTINGFILENAME, 0.33)
training_df, testing_df = pd.read_csv(TRANIINGFILENAME), pd.read_csv(TESTINGFILENAME)
training_df, testing_df = training_df.drop(training_df.columns[0], axis = 1), testing_df.drop(testing_df.columns[0], axis = 1)

# KNN CLASSIFIER WITH LIB
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def calcMacrosMicros(cmatrix):
    precision, recall = [], []
    coltotals, rowtotals, diagonalvals = cmatrix.sum(axis=0).tolist(), cmatrix.sum(axis=1).tolist(), cmatrix.diagonal().tolist()
    for i in range(len(diagonalvals)):
        precision.append((diagonal_val:=diagonalvals[i])/rowtotals[i])
        if(coltotals[i] == 0): recall.append(0)                     #there may be instances which our model doesn't predict at all (especially in imbalanced datasets); thus we may have a coltotal value of 0
        else: recall.append((diagonal_val/coltotals[i]))
    
    return sum(precision)/len(precision), sum(recall)/len(recall), sum(diagonalvals)/sum(rowtotals), sum(diagonalvals)/sum(coltotals)           #returns macprec, macrecall, micprec, micrecall

def knn_lib(training_df, testing_df, k_num):
    print("KNN LIB")
    x_train, y_train = training_df.iloc[:,[i for i in range(len(training_df.columns)-1)]].to_numpy(), training_df[training_df.columns[len(training_df.columns)-1]].to_numpy()
    x_test, y_test = testing_df.iloc[:,[i for i in range(len(testing_df.columns)-1)]].to_numpy(), testing_df[testing_df.columns[len(testing_df.columns)-1]].to_numpy()

    knn_clf = KNeighborsClassifier(n_neighbors=k_num)
    knn_clf.fit(x_train, y_train)

    y_pred = knn_clf.predict(x_test)

    acc_score = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred, labels = LABELPVALS)
    macroPrecision, macroRecall, microPrecision, microRecall = calcMacrosMicros(conf_mat)

    print(f"Testing Accuracy: {acc_score}")
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(conf_mat)
    print(f"Macroaverage Precision: {macroPrecision},   Macroaverage Recall: {macroRecall}")
    print(f"Microaverage Precision: {microPrecision},   Microaverage Recall: {microRecall}")

knn_lib(training_df, testing_df, K_NEIGHBORS)

# TEST OUT DiFF K VALS
import matplotlib.pyplot as plt

def knnLib(training_df, testing_df, k_num):
    x_train, y_train = training_df.iloc[:,[i for i in range(len(training_df.columns)-1)]].to_numpy(), training_df[training_df.columns[len(training_df.columns)-1]].to_numpy()
    x_test, y_test = testing_df.iloc[:,[i for i in range(len(testing_df.columns)-1)]].to_numpy(), testing_df[testing_df.columns[len(testing_df.columns)-1]].to_numpy()

    knn_clf = KNeighborsClassifier(n_neighbors=k_num)
    knn_clf.fit(x_train, y_train)

    y_pred = knn_clf.predict(x_test)

    return accuracy_score(y_test, y_pred)

def testK_knn(training_df, testing_df, kvalLB, kvalUB):
    x, y = [i for i in range(kvalLB, kvalUB)], []
    for kval in x: y.append(knnLib(training_df, testing_df, kval))
    plt.scatter(x, y, s=5, alpha=0.5)
    plt.show()
    input()

#testK_knn(training_df, testing_df, 1, 50)