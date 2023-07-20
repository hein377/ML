#Global variables
from sre_parse import State


ls = []
a1, a2, a3, a4, a5 = [], [], [], [], []

#Set-up
with open("a3-p2.csv") as f:
    for line in f.readlines()[1:]:
        l = line.split(',')
        for attribute_val in l: ls.append(int(attribute_val))

length = len(ls)
a1 = [ls[i] for i in range(0, length, 5)]
a2 = [ls[i] for i in range(1, length, 5)]
a3 = [ls[i] for i in range(2, length, 5)]
a4 = [ls[i] for i in range(3, length, 5)]
a5 = [ls[i] for i in range(4, length, 5)]

#Methods
def mean(ls): return sum(ls) / len(ls)

def variance(ls): 
    m = mean(ls)
    return sum([(x-m)**2 for x in ls]) / len(ls)

def standard_deviation(ls): return variance(ls)**0.5

def covariance(x, y):                                                                                                 #x, y are lists of same size
    mx, my = mean(x), mean(y)
    return (1/len(x)) * sum([(x[i]-mx)*(y[i]-my) for i in range(len(x))])

def correlation(x, y): return (covariance(x, y))/(standard_deviation(x)*standard_deviation(y))                 #x, y are lists of same size

#Output
print(f"correl(A1, A5): {correlation(a1, a5)}")
print(f"correl(A2, A5): {correlation(a2, a5)}")
print(f"correl(A3, A5): {correlation(a3, a5)}")
print(f"correl(A4, A5): {correlation(a4, a5)}")