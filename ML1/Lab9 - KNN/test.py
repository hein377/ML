def calcMacrosMicros(cmatrix):
    precision, recall = [], []
    coltotals, rowtotals, diagonalvals = cmatrix.sum(axis=0).tolist(), cmatrix.sum(axis=1).tolist(), cmatrix.diagonal().tolist()
    for i in range(len(diagonalvals)):
        precision.append((diagonal_val:=diagonalvals[i])/rowtotals[i])
        recall.append((diagonal_val/coltotals[i]))
    
    return sum(precision)/len(precision), sum(recall)/len(recall), sum(diagonalvals)/sum(rowtotals), sum(diagonalvals)/sum(coltotals)           #returns macprec, macrecall, micprec, micrecall

import numpy as np
cmatrix = np.zeros((3,3))
cmatrix[0,0] = 8
cmatrix[0,1] = 10
cmatrix[0,2] = 1
cmatrix[1,0] = 5
cmatrix[1,1] = 60
cmatrix[1,2] = 50
cmatrix[2,0] = 3
cmatrix[2,1] = 30
cmatrix[2,2] = 200
print(cmatrix)
macprec, macrec, micprec, micrec = calcMacrosMicros(cmatrix)
print(f"Macroaverage Precision: {macprec},   Macroaverage Recall: {macrec}")
print(f"Microaverage Precision: {micprec},   Microaverage Recall: {micrec}")

s = r'(Carnegie Mellon)|(University of Washington)|(MIT)|(University of California)|(Berkeley)|(UCLA)|(Champlain)|(GeorgiaTech)|(GTech)|(Georgia)|(Purdue)|(Stanford)|(University of Chicago)|(UChicago)|(Colorado Boulder)|(CUBoulder)|(Illinois Urbana-Champaign)|(UIUC)|(UMaryland)|(University of Maryland)|(Yale)|(Virginia Tech)|(VATech)'
s = s.replace(r'(', r'+"')
s = s.replace(r')', r'"')
print(s)