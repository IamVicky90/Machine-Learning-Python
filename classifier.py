from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris=datasets.load_iris()
features=iris.data
# print("f",features)
labels=iris.target
# print(features[0],labels[0])
clf=KNeighborsClassifier()
clf.fit(features,labels)
predicts=clf.predict([[2,90,4,5]])
print(predicts)