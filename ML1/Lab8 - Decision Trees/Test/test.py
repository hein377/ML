from anytree import Node, RenderTree
'''root = Node("root", children=[
     Node("studied:F", children=[
         Node("GPA:L"),
         Node("GPA:M"),
         Node("GPA:H")
     ]),
     Node("studied:T")
])

for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))

ls = [Node("studied:F"), Node("studied:T")]
root1 = Node("root")
root1.children = ls
for pre, fill, node in RenderTree(root1):
    print("%s%s" % (pre, node.name))'''

import pandas as pd
import math
import json

df = pd.read_csv("exampledf.csv")
IRIS_PVAL = {"gpa": ["l", "m", "h"], "studied": ["f", "t"], "passed": ["f", "t"]}

def pos_values(s): return list(set(s))

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
    s, pval_freq, total = "", {}, len(label_col)
    for labelval in label_col: add_dic(pval_freq, labelval)
    for val,freq in pval_freq.items(): pval_freq[val] = pval_freq[val]/total        #now pval_freq = pval_prob
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

tree_model=recur(Node("root"), df, set(df.columns[:-1]), 0)
for pre, fill, node in RenderTree(tree_model):
    print("%s%s" % (pre, node.name))

print()
in1 = {"gpa":"l", "studied":"t", "passed":"t"}
in2 = {"gpa":"m", "studied":"f", "passed":"f"}
in3 = {"gpa":"h", "studied":"f", "passed":"t"}
root = tree_model.root

import random

def choose_label(label_prob, n):
    return random.choices(list(label_prob.keys()), weights=label_prob.values(), k=n)

label_prob = {"a": 1/3, "b":1/2, "c":1/6}
choicesls = choose_label(label_prob, 100)
print(choicesls)
print(choicesls.count("a"))
print(choicesls.count("b"))
print(choicesls.count("c"))

for i in range(100): 
    print(random.randint(0, 2), end=" ")