import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris=datasets.load_iris()
# iris=list(iris)
x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)
# y=iris.target
model =LogisticRegression()
model.fit(x,y)
y_predict=model.predict([[9.6]])
x_new=np.linspace(0,3,1000).reshape(-1,1)
y_prob=model.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1])
plt.show()
print(y_prob[:,1])
# print(y_prob)