'''from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

circle_X, circle_y = make_circles(1000, noise=0.068, factor=0.8)
circle_X_train, circle_X_test, circle_y_train, circle_y_test = train_test_split(circle_X, circle_y, test_size=0.3, random_state=1, stratify = circle_y)
print(circle_y_train)
print(circle_y_train[0])'''

import neural_network_instantiation
network = neural_network_instantiation.create_network([2, 5, 5, 1], -1, 1)
print(len(network))