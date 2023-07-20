import pandas as pd
import numpy as np

def pos_values(s): return list(set(s))

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
    n, label_num = len(ls:=(pvals[labelname])), {"dog":0, "cat":1}
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
    print(pd.DataFrame(cmatrix, columns=col_labels, index=row_labels), end = "\n\n")

training_df = pd.read_csv("catdogexampletrain.csv")
testing_df = pd.read_csv("catdogexampletest.csv")
training_df, pvals  = binning(training_df, 3)
naiveBayes(training_df, testing_df, pvals)