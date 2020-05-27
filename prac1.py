import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model
diabates=datasets.load_diabetes()
# print(diabates.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabates.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
diabates_x=diabates.data[:,np.newaxis,2]
diabates_x_train=diabates_x[:-30]
diabates_x_test=diabates_x[-30:]
print(diabates_x_test)