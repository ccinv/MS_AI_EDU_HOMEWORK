
import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
 
iris = datasets.load_iris()
 
train_data, test_data, train_label, test_label = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

mlp = MLPClassifier(hidden_layer_sizes=(20,3))
mlp.fit(train_data,train_label)
 
pre = mlp.predict(test_data)
print("Training set score: %f" % mlp.score(train_data, train_label))
print("Test set score: %f" % mlp.score(test_data, test_label))

print(test_label)
print(pre)
