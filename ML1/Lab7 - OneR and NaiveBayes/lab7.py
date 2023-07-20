import numpy as np
import pandas as pd
from tabulate import tabulate
from random import randrange

# PROCESS FILE
FILENAME, TRANIINGFILENAME, TESTINGFILENAME = "lab7_iris.csv", "lab7_iris_pytraining.txt", "lab7_iris_pytesting.txt"
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

iris_df, iris_pvals  = binning(iris_df, 3)

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

#training_df, testing_df = stratified_random_sampling(iris_df, "lab7_iris_prettytable.txt", TRANIINGFILENAME, TESTINGFILENAME, 0.33)
training_df, testing_df = pd.read_csv(TRANIINGFILENAME), pd.read_csv(TESTINGFILENAME)
training_df, testing_df = training_df.drop(training_df.columns[0], axis = 1), testing_df.drop(testing_df.columns[0], axis = 1)

# ONER CLASSIFICATION
def dicadd(dic, key, labelval):
    if(labelval in dic[key]): dic[key][labelval] += 1
    else: dic[key].update({labelval : 1})

def find_bestmodel(df, models, attributename_pvals):
    models_accRate = []
    for i, (name, pvals) in enumerate(attributename_pvals.items()):
        if(name != df.columns[-1]):
            attribute_bestmodel, totalnumcorrect, totalnum = [], 0, 0
            for val in pvals:
                label_freq = models[i][val]
                total, bestlabel, bestfreq = 0, "", 0
                for label, freq in label_freq.items():
                    if freq>bestfreq: bestlabel, bestfreq = label, freq
                    total += freq
                attribute_bestmodel.append((val,bestlabel))
                totalnumcorrect += bestfreq
                totalnum += total
            models_accRate.append({tuple(attribute_bestmodel):totalnumcorrect/totalnum})

    bestmodel, bestaccRate = (), 0
    for model_accRate in models_accRate:
        if((accRate:=model_accRate[(model:=list(model_accRate.keys())[0])]) > bestaccRate): bestmodel, bestaccRate = model, accRate

    bestmodeldic = {}
    for tup in bestmodel:
        attributeval, labelval = tup
        bestmodeldic.update({attributeval : labelval})

    return bestmodeldic, bestaccRate

def createoneRmodel(df, attributename_pvals):         #df = training_df, pvals = {attributeName : [possible_values (floats or strings)]}
    models, instance_data = [{} for i in range(len(df.columns)-1)], training_df.iloc[:,[i for i in range(len(training_df.columns))]].to_numpy()                          #[{(attribute1_val,label_val) : freq (int)}, {(attribute2_val,label_val) : freq (int)} ...] e.g. { (sunny, No) : 5, (hot, Yes) : 2, ... }
    for i, (name, pvals) in enumerate(attributename_pvals.items()):
        if(name != df.columns[-1]):
            for val in pvals: models[i].update({val:{}})
        
    for instance in instance_data:
        labelval = instance[-1]
        for i in range(len(instance)-1): dicadd(models[i], instance[i], labelval)

    return find_bestmodel(df, models, attributename_pvals)       #models = [ { a1val1: {labelval : freq}, a1val2: {labelval:freq} }, { a2val1: {labelval : freq}, a2val2: {labelval:freq}, ... } ]

def testoneRmodel(model, testingdf, pvals):
    attributepvals, attributename, labelname = list(model.keys()), "", testingdf.columns[-1]
    n, label_num = len(ls:=(pvals[labelname])), {"Iris-versicolor":0, "Iris-virginica":1, "Iris-setosa":2}
    cmatrix = np.zeros((n, n))

    for name,pvals in pvals.items():
        if pvals == attributepvals: attributename = name
    #for i in range(n):
    #    label_num.update({ls[i]:i})

    for i in range(len(testingdf)):
        attval, labelval = testingdf.loc[i, attributename], testingdf.loc[i, labelname]
        model_prediction = model[attval]
        x, y = label_num[labelval], label_num[model_prediction]         #(actual, prediction)
        cmatrix[x,y] += 1

    return cmatrix, label_num, (np.trace(cmatrix)/np.sum(cmatrix))             # accrate = sum(diagonal)/sum(matrix elements)

def oneR(trainingdf, testingdf, pvals):
    model, training_accRate = createoneRmodel(trainingdf, pvals)         #model = [ {attribute1val:labelval}, {attribute2val:labelval}, ... ]
    cmatrix, label_num, testing_accRate = testoneRmodel(model, testingdf, pvals)
    print("ONER")
    print(f"ONER model rules: {model}")
    print(f"Testing Accuracy: {testing_accRate}")
    row_labels = col_labels = list(label_num.keys())
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(pd.DataFrame(cmatrix, columns=col_labels, index=row_labels), end = "\n\n\n")

oneR(training_df, testing_df, iris_pvals)

# NAIVEBAYES CLASSIFICATION
def addDic(dic, key):
    if key in dic: dic[key]+=1
    else: dic.update({key:1})

def createNBModel(df, pvals):
    numrows, label_count, attlabel_count, pLabels, pConditional = df.shape[0], {}, {}, {}, {}
    labelpvals = pvals[df.columns[-1]]
    for attName, attpvals in pvals.items():
        if(attName != df.columns[-1]):
            for attval in attpvals:
                for labelval in labelpvals: attlabel_count.update({(attval, labelval):0})

    for index, row in df.iterrows():
        addDic(label_count, (labelval:=row[-1]))
        for i in range(len(row)-1): addDic(attlabel_count, (row[i], labelval))
    
    for label,count in label_count.items(): pLabels.update({label:(count/numrows)})
    for attlabel,count in attlabel_count.items(): 
        attval, labelval = attlabel
        pConditional.update({attlabel:(count/label_count[labelval])})

    return pLabels, pConditional

def chooseHighestProbLabel(label_prob):
    bestlabel, bestprob = "", 0
    for label,prob in label_prob.items():
        if prob>bestprob: bestlabel, bestprob = label, prob
    return bestlabel

def findProb(labelval, instance, pLabels, pConditional):
    prob = pLabels[labelval]
    for i in range(len(instance)-1):
        prob *= pConditional[(instance[i],labelval)]
    return prob

def classifyingInstanceNB(instance, pLabels, pConditional, pvals, labelname):
    label_prob = {}
    for labelval in pvals[labelname]: label_prob.update({labelval:findProb(labelval, instance, pLabels, pConditional)})
    return chooseHighestProbLabel(label_prob)

def testNBModel(df, pLabels, pConditional, pvals):
    labelname = df.columns[-1]
    n, label_num = len(ls:=(pvals[labelname])), {"Iris-versicolor":0, "Iris-virginica":1, "Iris-setosa":2}
    cmatrix = np.zeros((n, n))
    
    #for i in range(n):
    #    label_num.update({ls[i]:i})

    for index, row in df.iterrows():
        true_labelval, model_prediction = row[-1], classifyingInstanceNB(row, pLabels, pConditional, pvals, labelname)
        x, y = label_num[true_labelval], label_num[model_prediction]
        cmatrix[x,y] += 1

    return cmatrix, label_num, (np.trace(cmatrix)/np.sum(cmatrix))             # accrate = sum(diagonal)/sum(matrix elements)

def naiveBayes(trainingdf, testingdf, pvals):
    pLabels, pConditional = createNBModel(trainingdf, pvals)
    cmatrix, label_num, testing_accRate = testNBModel(testingdf, pLabels, pConditional, pvals)
    print("NAIVEBAYES", end="\n")
    print(f"Testing Accuracy: {testing_accRate}")
    row_labels = col_labels = list(label_num.keys())
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(pd.DataFrame(cmatrix, columns=col_labels, index=row_labels), end = "\n\n\n")

naiveBayes(training_df, testing_df, iris_pvals)

# ONER CLASSIFICATION WITH LIBRARY
#http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.classifier/
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from mlxtend.classifier import OneRClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def oneR_lib(training_df, testing_df, pvals):
    print("ONER LIB")
    x_train, y_train = training_df.iloc[:,[i for i in range(len(training_df.columns)-1)]].to_numpy(), training_df[training_df.columns[len(training_df.columns)-1]].to_numpy()
    x_test, y_test = testing_df.iloc[:,[i for i in range(len(testing_df.columns)-1)]].to_numpy(), testing_df[testing_df.columns[len(testing_df.columns)-1]].to_numpy()

    # label_num, num_label = {}, {}
    # for i in range(n):
    #    label_num.update({ls[i]:i})
    #    num_label.update({i:ls[i]})
    label_num = {"Iris-versicolor":0, "Iris-virginica":1, "Iris-setosa":2}
    num_label = {0:"Iris-versicolor", 1:"Iris-virginica", 2:"Iris-setosa"}
    y_train = np.array([label_num[y_train[i]] for i in range(len(y_train))])
    y_test = np.array([label_num[y_test[i]] for i in range(len(y_test))])

    oneR_clf = OneRClassifier()
    oneR_clf.fit(x_train, y_train)

    y_pred = oneR_clf.predict(x_test)

    y_train = np.array([num_label[y_train[i]] for i in range(len(y_train))])
    y_test = np.array([num_label[y_test[i]] for i in range(len(y_test))])
    y_pred = np.array([num_label[y_pred[i]] for i in range(len(y_pred))])

    acc_score = accuracy_score(y_test, y_pred)
    #labelpvals = pvals[training_df.columns[-1]]
    labelpvals = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]
    conf_mat = confusion_matrix(y_test, y_pred, labels = labelpvals)

    print(f"Testing Accuracy: {acc_score}")
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(conf_mat)
    print("\n")

oneR_lib(training_df, testing_df, iris_pvals)

# NAIVEBAYES CLASSIFICATION WITH LIBRARY
#https://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.naive_bayes import CategoricalNB

def naiveBayes_lib(training_df, testing_df, pvals):
    print("NAIVEBAYES LIB")
    x_train, y_train = training_df.iloc[:,[i for i in range(len(training_df.columns)-1)]].to_numpy(), training_df[training_df.columns[len(training_df.columns)-1]].to_numpy()
    x_test, y_test = testing_df.iloc[:,[i for i in range(len(testing_df.columns)-1)]].to_numpy(), testing_df[testing_df.columns[len(testing_df.columns)-1]].to_numpy()

    nb_clf = CategoricalNB()                #Categorical because we're inputting binned data (GaussianNB() if unbinned data)
    nb_clf.fit(x_train, y_train)

    y_pred = nb_clf.predict(x_test)

    acc_score = accuracy_score(y_test, y_pred)
    #labelpvals = pvals[training_df.columns[-1]]
    labelpvals = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]
    conf_mat = confusion_matrix(y_test, y_pred, labels = labelpvals)

    print(f"Testing Accuracy: {acc_score}")
    print("Confusion Matrix (columns=predictions, rows=actualVals):")
    print(conf_mat)

naiveBayes_lib(training_df, testing_df, iris_pvals)