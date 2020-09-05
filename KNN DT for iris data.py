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

iris = datasets.load_iris()
X = iris.data[:,2:4]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=33, stratify=y)

# Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
#按垂直方向叠加 有点迷
y_combined = np.hstack((y_train, y_test))
#按水平方向平铺 有点迷

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

## Building a decision tree
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()

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
print('DecisionTree:Best t>=',max_index+1,'TestAcurracy=',maxknn)
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(krange, testscores, label = 'Testing Accuracy')
plt.plot(krange,  trainscores, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


#create an image (老师上课说这里函数有点问题，自己想办法)

dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=['Setosa', 
                                        'Versicolor',
                                        'Virginica'],
                           feature_names=['petal length', 
                                          'petal width'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('tree.png') 


# # K-nearest neighbors - a lazy learning algorithm
knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)
#print(knn.score(X_test_std, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_24.png', dpi=300)
plt.show()

'''
#test the best k before std
neighbors = np.arange(1,26)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    train_accuracy[i]=knn.score(X_train_std, y_train)
    test_accuracy[i]=knn.score(X_test_std, y_test)
    
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
'''
#test the best k second version after std
krange=range(1,26)
testscores=[]
trainscores=[]
for k in krange:
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    y_pred=knn.predict(X_test_std)
    y_train_pred = knn.predict(X_train_std)
    testscores.append(metrics.accuracy_score(y_test,y_pred))
    trainscores.append(metrics.accuracy_score(y_train,y_train_pred))
#print best k and scores
maxknn=max(testscores)
max_index=testscores.index(maxknn)
print('KNN:Best k=4,5,7','TestAcurracy=',maxknn)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(krange, testscores, label = 'Testing Accuracy')
plt.plot(krange,  trainscores, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

print("My name is {Xin Zhang}")
print("My NetID is: {xzhan81}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")