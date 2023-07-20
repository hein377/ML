import numpy as np
import pandas as pd
from tabulate import tabulate
from random import randrange
from random import randint
from matplotlib import pyplot as plt
import seaborn as sns

# FILE SET UP
from sklearn.model_selection import train_test_split
def make_skewed_dataset(att_info):
    x, y = np.zeros((total_instances:=sum([list(att_info.values())[i][0] for i in range(len(att_info))]),len(list(att_info.values())[0])-1)), [0]*total_instances
    row_ind = 0
    for label_name, val in att_info.items():
        num_instances, attribute_ranges = val[0], val[1:]
        for row in range(row_ind, row_ind+num_instances):
            y[row] = label_name
            for col in range(len(attribute_ranges)):
                lb, ub = attribute_ranges[col]
                x[row][col] = randint(lb, ub)
        row_ind += num_instances
    return x, y

def make_cols_floats(df): 
    for i in range(len(df.columns)-1): df[df.columns[i]] = df[df.columns[i]].astype(float)
    return df

x, y = make_skewed_dataset({"label0":[20, (10,20), (10,20)], "label1":[230, (15, 30), (15, 30)], "label2":[80, (10, 20), (20, 30)]})          #x=attribute_matrix, y=label_col
SKEWED_DF = make_cols_floats(pd.DataFrame(np.column_stack((x,y)), columns=['att1', 'att2', 'label']))

#Code for Scatterplot of Skewed Dataset:
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.scatterplot(data=SKEWED_DF, x='att1', y='att2', hue='label')

plt.title("att1 vs. att2")
plt.xlabel("att1")
plt.ylabel("att2")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
skewed_trainingdf, skewed_testingdf = make_cols_floats(pd.DataFrame(np.column_stack((x_train,y_train)), columns=['att1', 'att2', 'label'])), make_cols_floats(pd.DataFrame(np.column_stack((x_test,y_test)), columns=['att1', 'att2', 'label']))

def pos_values(s): 
    pvals = list(set(s))
    pvals.sort()
    return pvals

def print_pretty_table(dataset, filename):
    f = open(filename, "w")
    f.write(tabulate(dataset, headers='keys', tablefmt='psql'))
    f.close()

def get_attpvals(df):
    att_pvals = {}
    for (attributeName, colData) in df.iteritems(): att_pvals.update({attributeName:pos_values(colData)})
    return att_pvals

ATT_PVALS = get_attpvals(SKEWED_DF)
LABELPVALS = ATT_PVALS[SKEWED_DF.columns[-1]]
LABEL_NUM, NUM_LABEL = {}, {}
for i in range(len(LABELPVALS)):
    LABEL_NUM.update({LABELPVALS[i]:i})
    NUM_LABEL.update({i:LABELPVALS[i]})
K_NEIGHBORS = 5

# KNN CLASSIFIER
from scipy.spatial.distance import euclidean
def add_dic(dic, key):
    if(key in dic): dic[key] += 1
    else: dic.update({key:1})

def knn_choose_best(sorted_ls):         #sorted_ls = [ instance1<tup>, instance2<tup>, ... , instancek<tup> ] sorted by smallest to largest distance
    labelval_freq = {}
    for instance in sorted_ls: add_dic(labelval_freq, instance[-1])
    return sorted(labelval_freq, key = lambda x: labelval_freq[x], reverse = True)[0]           #return class label value with the highest frequency

def knn_classify(training_df, testing_instance, k_num):                  #testing_instance <tup>
    traininginstance_distance = {}
    for index, training_instance in training_df.iterrows():
        training_instance = tuple(training_instance)
        if(training_instance not in traininginstance_distance): traininginstance_distance.update({training_instance:euclidean(training_instance[:-1], testing_instance[:-1])})               #instance[-1] to ensure class label value is not used to calculate euclidean distance
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
        if(coltotals[i] == 0): recall.append(0)
        else: recall.append((diagonal_val/coltotals[i]))
    
    return sum(precision)/len(precision), sum(recall)/len(recall), sum(diagonalvals)/sum(rowtotals), sum(diagonalvals)/sum(coltotals)           #returns macprec, macrecall, micprec, micrecall

def knn(training_df, testing_df, k_num):
    cmatrix, testing_accRate = knn_test(training_df, testing_df, k_num)
    macroPrecision, macroRecall, microPrecision, microRecall = calcMacrosMicros(cmatrix)

    print("KNN", end="\n")
    print(f"Testing Accuracy: {testing_accRate}")
    row_labels = col_labels = list(LABEL_NUM.keys())
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(pd.DataFrame(cmatrix, columns=col_labels, index=row_labels))
    print(f"Macroaverage Precision: {macroPrecision},   Macroaverage Recall: {macroRecall}", end = "\n")
    print(f"Microaverage Precision: {microPrecision},   Microaverage Recall: {microRecall}", end = "\n")

# KNN CLASSIFIER WITH LIB
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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
    print(f"Macroaverage Precision: {macroPrecision},   Macroaverage Recall: {macroRecall}", end = "\n")
    print(f"Microaverage Precision: {microPrecision},   Microaverage Recall: {microRecall}", end = "\n")

# NAIVEBAYES CLASSIFICATION WITH LIBRARY
#https://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.naive_bayes import CategoricalNB

def naiveBayes_lib(training_df, testing_df):
    print("NAIVEBAYES LIB")
    x_train, y_train = training_df.iloc[:,[i for i in range(len(training_df.columns)-1)]].to_numpy(), training_df[training_df.columns[len(training_df.columns)-1]].to_numpy()
    x_test, y_test = testing_df.iloc[:,[i for i in range(len(testing_df.columns)-1)]].to_numpy(), testing_df[testing_df.columns[len(testing_df.columns)-1]].to_numpy()

    nb_clf = CategoricalNB()                #Categorical because we're inputting binned data (GaussianNB() if unbinned data)
    nb_clf.fit(x_train, y_train)

    y_pred = nb_clf.predict(x_test)

    acc_score = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred, labels = LABELPVALS)
    macroPrecision, macroRecall, microPrecision, microRecall = calcMacrosMicros(conf_mat)

    print(f"Testing Accuracy: {acc_score}")
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(conf_mat)
    print(f"Macroaverage Precision: {macroPrecision},   Macroaverage Recall: {macroRecall}", end = "\n")
    print(f"Microaverage Precision: {microPrecision},   Microaverage Recall: {microRecall}", end = "\n")

# ONER CLASSIFICATION WITH LIBRARY
#http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.classifier/
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from mlxtend.classifier import OneRClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def oneR_lib(training_df, testing_df):
    print("ONER LIB")
    x_train, y_train = training_df.iloc[:,[i for i in range(len(training_df.columns)-1)]].to_numpy(), training_df[training_df.columns[len(training_df.columns)-1]].to_numpy()
    x_test, y_test = testing_df.iloc[:,[i for i in range(len(testing_df.columns)-1)]].to_numpy(), testing_df[testing_df.columns[len(testing_df.columns)-1]].to_numpy()

    y_train = np.array([LABEL_NUM[y_train[i]] for i in range(len(y_train))])
    y_test = np.array([LABEL_NUM[y_test[i]] for i in range(len(y_test))])

    oneR_clf = OneRClassifier()
    oneR_clf.fit(x_train, y_train)

    y_pred = oneR_clf.predict(x_test)

    y_train = np.array([NUM_LABEL[y_train[i]] for i in range(len(y_train))])
    y_test = np.array([NUM_LABEL[y_test[i]] for i in range(len(y_test))])
    y_pred = np.array([NUM_LABEL[y_pred[i]] for i in range(len(y_pred))])

    acc_score = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred, labels = LABELPVALS)
    macroPrecision, macroRecall, microPrecision, microRecall = calcMacrosMicros(conf_mat)

    print(f"Testing Accuracy: {acc_score}")
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(conf_mat)
    print(f"Macroaverage Precision: {macroPrecision},   Macroaverage Recall: {macroRecall}", end = "\n")
    print(f"Microaverage Precision: {microPrecision},   Microaverage Recall: {microRecall}", end = "\n")

# DECISION TREE CLASSIFICATION WITH LIBRARY
#https://scikit-learn.org/stable/modules/tree.html
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def tree_lib(training_df, testing_df):
    print("DECISIONTREES LIB")
    x_train, y_train = training_df.iloc[:,[i for i in range(len(training_df.columns)-1)]].to_numpy(), training_df[training_df.columns[len(training_df.columns)-1]].to_numpy()
    x_test, y_test = testing_df.iloc[:,[i for i in range(len(testing_df.columns)-1)]].to_numpy(), testing_df[testing_df.columns[len(testing_df.columns)-1]].to_numpy()

    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(x_train, y_train)

    y_pred = tree_clf.predict(x_test)

    acc_score = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred, labels = LABELPVALS)
    macroPrecision, macroRecall, microPrecision, microRecall = calcMacrosMicros(conf_mat)

    print(f"Testing Accuracy: {acc_score}")
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(conf_mat)
    print(f"Macroaverage Precision: {macroPrecision},   Macroaverage Recall: {macroRecall}", end = "\n")
    print(f"Microaverage Precision: {microPrecision},   Microaverage Recall: {microRecall}", end = "\n")

# DECISION TREE - RANDOMFOREST CLASSIFICATION WITH LIB
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def randomforest_lib(training_df, testing_df):
    print("RANDOMFOREST LIB")
    x_train, y_train = training_df.iloc[:,[i for i in range(len(training_df.columns)-1)]].to_numpy(), training_df[training_df.columns[len(training_df.columns)-1]].to_numpy()
    x_test, y_test = testing_df.iloc[:,[i for i in range(len(testing_df.columns)-1)]].to_numpy(), testing_df[testing_df.columns[len(testing_df.columns)-1]].to_numpy()

    tree_clf = RandomForestClassifier(n_estimators = 100) 
    tree_clf.fit(x_train, y_train)

    y_pred = tree_clf.predict(x_test)

    acc_score = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred, labels = LABELPVALS)
    macroPrecision, macroRecall, microPrecision, microRecall = calcMacrosMicros(conf_mat)

    print(f"Testing Accuracy: {acc_score}")
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(conf_mat)
    print(f"Macroaverage Precision: {macroPrecision},   Macroaverage Recall: {macroRecall}", end = "\n")
    print(f"Microaverage Precision: {microPrecision},   Microaverage Recall: {microRecall}", end = "\n")

# TESTING
print("METHODS W/O BINNING:")
knn(skewed_trainingdf, skewed_testingdf, K_NEIGHBORS)
knn_lib(skewed_trainingdf, skewed_testingdf, K_NEIGHBORS)
print("\n")
naiveBayes_lib(skewed_trainingdf, skewed_testingdf)
oneR_lib(skewed_trainingdf, skewed_testingdf)
tree_lib(skewed_trainingdf, skewed_testingdf)
randomforest_lib(skewed_trainingdf, skewed_testingdf)
print("\n")

def binning(df, n):                            #df = dataframe; n = # of bins; equal width binning for quantitative attributes
    pvals = {}
    for (attributeName, colData) in df.iteritems():
        if(colData.dtype != object):
            bin_width = ((maximum:=max(colData)) - (minimum:=min(colData)))/n
            bins = [round(minimum+i*bin_width, 3) for i in range(n+1)]
            labels = [round((bins[i]+bins[i+1])/2, 3) for i in range(n)]
            df[attributeName] = pd.cut(df[attributeName],bins,labels=labels,include_lowest=True)
            pvals.update({attributeName:labels})
        else: pvals.update({attributeName:pos_values(colData)})
    return df.to_numpy(), pvals

binned_skewed_df, binned_skeweddf_pvals  = binning(SKEWED_DF, 3)
x, y = binned_skewed_df[:,[i for i in range(binned_skewed_df.shape[1]-1)]], binned_skewed_df[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
skewed_trainingdf, skewed_testingdf = make_cols_floats(pd.DataFrame(np.column_stack((x_train,y_train)), columns=list(binned_skeweddf_pvals.keys()))), make_cols_floats(pd.DataFrame(np.column_stack((x_test,y_test)), columns=list(binned_skeweddf_pvals.keys())))

print("METHODS W/ BINNING:")
knn(skewed_trainingdf, skewed_testingdf, K_NEIGHBORS)
knn_lib(skewed_trainingdf, skewed_testingdf, K_NEIGHBORS)
print("\n")
naiveBayes_lib(skewed_trainingdf, skewed_testingdf)
oneR_lib(skewed_trainingdf, skewed_testingdf)
tree_lib(skewed_trainingdf, skewed_testingdf)
randomforest_lib(skewed_trainingdf, skewed_testingdf)