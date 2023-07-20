import numpy as np
import pandas as pd
from tabulate import tabulate
import random
from random import randrange
import math
import json

# FILE SET UP
FILENAME, TRANIINGFILENAME, TESTINGFILENAME = "lab8_iris.csv", "lab8_iris_pytraining.txt", "lab8_iris_pytesting.txt"
iris_df = pd.read_csv(FILENAME)

def pos_values(s): return list(set(s))

def print_pretty_table(dataset, filename):
    f = open(filename, "w")
    f.write(tabulate(dataset, headers='keys', tablefmt='psql'))
    f.close()

# DATA PREPROCESSING
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
    return df, pvals

IRIS_DF, IRIS_PVAL  = binning(iris_df, 3)
#LABELPVALS = IRIS_PVAL[IRIS_DF.columns[-1]]
LABELPVALS = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]
LABEL_NUM, NUM_LABEL = {}, {}
for i in range(len(LABELPVALS)):
    LABEL_NUM.update({LABELPVALS[i]:i})
    NUM_LABEL.update({i:LABELPVALS[i]})

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

# DECISION TREE CLASSIFICATION
#https://anytree.readthedocs.io/en/latest/api/anytree.node.html#anytree.node.nodemixin.NodeMixin.children
#https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python
from anytree import Node, RenderTree

def print_tree(root):
    print("DECISIONTREE MODEL")
    for pre, fill, node in RenderTree(root): print("%s%s" % (pre, node.name))
    print()

def add_dic(dic, key):
    if(key in dic): dic[key] += 1
    else: dic.update({key:1})

def calc_entropy(ls):
    entropy, val_freq, total = 0, {}, len(ls)
    for val in ls: add_dic(val_freq, val)
    for val, freq in val_freq.items(): entropy += (freq/total)*math.log(freq/total,2)
    return -1*entropy

def create_newdf(df, attributename, val):
    att_ind, rows_toremove = list(IRIS_PVAL.keys()).index(attributename), []
    for row_ind, row in df.iterrows():
        if(row[att_ind] != val): rows_toremove.append(row_ind)
    df = df.drop(index=rows_toremove, axis=0)
    return df

def calc_weightedsum_entropy(attpval_entropy, dflen):
    weightedsum = 0
    for attpval_count,entropy in attpval_entropy.items():
        pval, count = attpval_count
        weightedsum += (count/dflen)*entropy
    return weightedsum

def calc_entropygain(df, attname, attpvals, cur_entropy):
    attpval_entropy = {}
    for attpval in attpvals: 
        new_df = create_newdf(df, attname, attpval)
        label_col = new_df[df.columns[-1]]
        attpval_entropy.update({(attpval, len(label_col)):calc_entropy(list(label_col))})       #attpval_entropy = {(attpval, newdf_length):entropy(newdf), ...}; e.g. if attribute = "gpa": {("l",2):1}
    return (cur_entropy - calc_weightedsum_entropy(attpval_entropy, len(list(df[df.columns[0]]))))

def choose_highestval_key(att_entropygain):
    return sorted(att_entropygain, key = lambda x: att_entropygain[x], reverse = True)[0]

def find_best_attribute(df, cur_entropy, feature_set):
    if(len(feature_set) == 1):
        return (attname:=next(iter(feature_set))), IRIS_PVAL[attname]       #if there is only 1 attribute left, return that attribute and its pvals
    att_entropygain = {}
    for attname in feature_set:
        attpvals = IRIS_PVAL[attname]
        if(attname != (labelname:=df.columns[-1])): att_entropygain.update({attname:calc_entropygain(df, attname, attpvals, cur_entropy)})
    return (attname:=choose_highestval_key(att_entropygain)), IRIS_PVAL[attname]

def calc_prob_labels(label_col):
    #https://docs.python.org/3/library/json.html
    s, pval_freq, total = "", {}, len(label_col)
    for labelval in label_col: add_dic(pval_freq, labelval)
    for val,freq in pval_freq.items(): pval_freq[val] = round(pval_freq[val]/total,3)        #now pval_freq = pval_prob
    return json.dumps(pval_freq)

def recur(root, df, feature_set, depth):
    if(df.empty or len(feature_set)==0 or (cur_entropy:=calc_entropy(list(df[df.columns[-1]].to_numpy())))==0):               #cur_entropy = entropy of last col = entropy of class label col
        root.name += "-->" + calc_prob_labels(df[df.columns[-1]])
        return root
    childrenls = []
    best_attribute, attribute_pvals = find_best_attribute(df, cur_entropy, feature_set)
    feature_set.remove(best_attribute)
    for val in attribute_pvals: childrenls.append(recur(Node(best_attribute+":"+str(val)), create_newdf(df, best_attribute, val), feature_set, depth+1))
    root.children = childrenls
    return root

def create_dtree_model(df):
    return recur(Node("root"), df, set(df.columns[:-1]), 0)

def choose_label(label_prob, n):
    #https://stackoverflow.com/questions/40927221/how-to-choose-keys-from-a-python-dictionary-based-on-weighted-probability
    if(label_prob == {}): return NUM_LABEL[random.randint(0,len(NUM_LABEL.keys())-1)]                    #if label_prob is empty; return a label w/ equal prob for each
    return random.choices(list(label_prob.keys()), weights=label_prob.values(), k=n)[0]

def treeclassify(tmodel, instance):                 #instance = {attname1:att1val, attname2: att2val, ...}
    root = tmodel.root
    while(len(cnodesls:=root.children) != 0):
        attname = (s:=cnodesls[0].name)[:s.index(":")]
        instanceAttVal = instance[attname]
        for cnode in cnodesls:
            if str(instanceAttVal) in cnode.name: 
                root = cnode
                break
    label_prob = json.loads((s:=root.name)[s.index(">")+1:])
    return choose_label(label_prob, 1)

def test_dtree_model(tmodel, df):
    cmatrix = np.zeros(((n:=len(LABEL_NUM.keys())), n))               #n=num_instances
    for index, row in df.iterrows():
        model_prediction, true_label_val = treeclassify(tmodel, row.to_dict()), row[-1]
        x, y = LABEL_NUM[true_label_val], LABEL_NUM[model_prediction]
        cmatrix[x,y] += 1
    return cmatrix, (np.trace(cmatrix)/np.sum(cmatrix))             # accrate = sum(diagonal)/sum(matrix elements)

def dtree(training_df, testing_df):
    dtree_model = create_dtree_model(training_df)
    #print_tree(dtree_model)
    cmatrix, testing_accRate = test_dtree_model(dtree_model, testing_df)

    print("DECISIONTREES", end="\n")
    print(f"Testing Accuracy: {testing_accRate}")
    row_labels = col_labels = list(LABEL_NUM.keys())
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(pd.DataFrame(cmatrix, columns=col_labels, index=row_labels), end = "\n\n\n")

dtree(training_df, testing_df)

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

    print(f"Testing Accuracy: {acc_score}")
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(conf_mat)

tree_lib(training_df, testing_df)