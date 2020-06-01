import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import load,dump

housing=pd.read_csv("data.csv")
print(housing.head())
print(housing.info())
print(housing.describe())
housing['CHAS'].value_counts()
housing['NOX'].value_counts()
housing.hist(bins=50,figsize=(20,15))
housing.hist(bins=10,figsize=(20,15))
def split_train_test_data(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indeces=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[test_indeces], data.iloc[train_indices]
test_set, train_set=split_train_test_data(housing,0.2)
print(len(train_set),len(test_set))


# from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(len(train_set),len(test_set))

# from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,random_state=42,test_size=0.2)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
strat_train_set['CHAS'].value_counts()
strat_test_set['CHAS'].value_counts()
strat_train_set.info()
strat_test_set.info()
housing=strat_train_set.copy() ## I do this because i have to work with the training data

corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes=['MEDV','RM','TAX','CHAS']
scatter_matrix(housing[attributes],figsize=(15,8))
housing.plot(kind="scatter",x="MEDV",y="RM",alpha=0.8)
plt.show()

attributes1=["MEDV","RM"]
scatter_matrix(housing[attributes1],figsize=(15,8))

#housing["RMEDV"]=housing["RM"]/housing["MEDV"]
housing.head()

new_corr_marrix=housing.corr()
new_corr_marrix["MEDV"].sort_values(ascending=False)
housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()
housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()

# Now you see that I remove 7 data points of RM
housing.describe()
# So to deal with these emplty points we have three options
# 1.Get rid of the missing data points
# 2.Get rid of the whole attribute in which the data points are missing if that data point is not important by corr() function
# 3. Set the value to some value(0,mean or median)
# Option 1:
a=housing.dropna()
a.shape
a=housing.dropna()
a.shape

# Option 2:
a=housing.drop("RM",axis=1)
a.describe()

#Third Option
median=housing["RM"].median()
median
housing["RM"].fillna(median)
housing.shape #Remember we still donot change the original housing to change it see the below process
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)
imputer.statistics_
imputer.statistics_.shape

X=imputer.transform(housing) 
###X.describe()  We could not describe it so to describe it we perform following process in next cell
housing_tr=pd.DataFrame(X,columns=housing.columns) ##Transform Data frame housing_tr means housing transform
housing_tr.describe() ##so we are able to describe it

# Very Important
# I added Scikit-learn Design as apicture named as Capture.PNG
# Must read it
# Must read Feature Scalling for pipe line

# Very Important
# I added Scikit-learn Design as apicture named as Capture.PNG
# Must read it
# Must read Feature Scalling for pipe line
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    #..... You can add so many pipelines
    ('std_scaller',StandardScaler()),
])
housing_num_tr=my_pipeline.fit_transform(housing)
housing_num_tr ## as it is an array so we can predict from it

housing_num_tr.shape

# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
some_data=housing.iloc[:5]
some_data
some_labels=housing_labels[:5]
prepared_data=my_pipeline.transform(some_data) ##To convert it into array
prepared_data
model.predict(prepared_data)
some_labels
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse
lin_mse

##To overcome overfitting use cross validation Score
#from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error")
rmse_scores=np.sqrt(-scores)
rmse_scores
# Saving the model
# from joblib import load,dump
dump(model,"Real_estate_py.joblib")

# Testing the model
X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
final_rmse

print(list(Y_test),list(final_predictions))

prepared_data[0]