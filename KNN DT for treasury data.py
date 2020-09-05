from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd

df = pd.read_csv('Treasury Squeeze test - DS1.csv', header=None)
X = df.values[1:900, 2:11]
y = df.values[1:900, 11]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33, stratify=y)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

# # K-nearest neighbors - a lazy learning algorithm
knn = KNeighborsClassifier(n_neighbors=25, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train, y_train)
#print(knn.score(X_test, y_test))
'''
#test the best k
krange=range(1,11)
testscores=[]
trainscores=[]
for k in krange:
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    y_train_pred = knn.predict(X_train)
    testscores.append(metrics.accuracy_score(y_test,y_pred))
    trainscores.append(metrics.accuracy_score(y_train,y_train_pred))

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(krange, testscores, label = 'Testing Accuracy')
plt.plot(krange,  trainscores, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
'''
#test the best k
krange=range(1,100)
testscores=[]
trainscores=[]
for k in krange:
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    y_train_pred = knn.predict(X_train)
    testscores.append(metrics.accuracy_score(y_test,y_pred))
    trainscores.append(metrics.accuracy_score(y_train,y_train_pred))
#print best k and scores
maxknn=max(testscores)
max_index=testscores.index(maxknn)
print('KNN:Best k=',max_index+1,'TestAcurracy=',maxknn)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(krange, testscores, label = 'Testing Accuracy')
plt.plot(krange,  trainscores, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

## Building a decision tree
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
tree.fit(X_train, y_train)

#test the best t
krange=range(1,11)
testscores=[]
trainscores=[]
for t in krange:
    tree = DecisionTreeClassifier(criterion='gini',  max_depth=t, random_state=1)
    tree.fit(X_train, y_train)
    y_pred=tree.predict(X_test)
    y_train_pred = tree.predict(X_train)
    testscores.append(metrics.accuracy_score(y_test,y_pred))
    trainscores.append(metrics.accuracy_score(y_train,y_train_pred))
#print best t and scores
maxknn=max(testscores)
max_index=testscores.index(maxknn)
print('DecisionTree:Best t=',max_index+1,'TestAcurracy=',maxknn)
# Generate plot
plt.title('Decision Tree: Varying Number of Neighbors')
plt.plot(krange, testscores, label = 'Testing Accuracy')
plt.plot(krange,  trainscores, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

print("My name is {Xin Zhang}")
print("My NetID is: {xzhan81}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")