import random
import numpy as np

def create_random_wm(layer1_size, layer2_size):                 #returns matrix of shape (layer2_size, layer1_size)
    ls = []
    for row in range(layer1_size): ls.append([random.uniform(-1, 1) for i in range(layer2_size)])
    return np.array(ls)

print(create_random_wm(3,2))

def create_random_bs(layer_size): return np.array([[random.uniform(-1, 1) for i in range(layer_size)]])

print(create_random_bs(3))