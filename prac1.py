import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
diabates=datasets.load_diabetes()
# print(diabates.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabates.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
diabates_x=diabates.data[:,np.newaxis,2]
diabates_x_train=diabates_x[:-30]
diabates_x_test=diabates_x[-30:]
diabates_y_train=diabates.target[:-30]
diabates_y_test=diabates.target[-30:]
model=linear_model.LinearRegression()
model.fit(diabates_x_train,diabates_y_train)
diabates_y_predict=model.predict(diabates_x_test)
print(f"meani square vales are:",mean_squared_error(diabates_y_test,diabates_y_predict))
print("weights: ",model.coef_)
print("slope: ", model.intercept_)
plt.scatter(diabates_x_test,diabates_y_test)
plt.plot(diabates_x_test,diabates_y_predict)
plt.show()
