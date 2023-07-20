import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
data = pd.read_csv("iris.csv")
setosa = data.head(50)
setosa_sepalw = setosa.sepalwidth.tolist()

plt.hist(setosa_sepalw, density=False, bins = 5)
plt.title('Iris-Setosa Sepal Width Distribution')
plt.ylabel('Frequency')
plt.xlabel('Sepal Width')
plt.show()