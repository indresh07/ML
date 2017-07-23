from sklearn.datasets import load_iris
import numpy as np
import random
from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a,b)

class KNN():
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []

		for row in x_test:
			predictions.append(self.closest(row))

		return predictions

	def closest(self, row):
		bestIndex = 0;
		minDist = euc(row, self.x_train[0])

		for i in range(1, len(self.x_train)):
			dist = euc(row, self.x_train[i])

			if dist < minDist:
				minDist = dist
				bestIndex = i;

		return self.y_train[bestIndex]


iris = load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

#DecisionTree
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

#Kneighbors
#from sklearn.neighbors import KNeighborsClassifier
clf = KNN()
clf.fit(X_train, Y_train)

results = clf.predict(X_test)

from  sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, results))

for result in results:
	print(iris.target_names[result])
