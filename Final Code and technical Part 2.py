#%% importing libraries
from os import remove
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

#%% reading csv as dataframe
df= pd.read_csv("Clean_Dataset.csv")
del df['Unnamed: 0']

# %% Preprocessing
#class
df['class'].replace(['Economy', 'Business'], [0, 1], inplace=True)

#arrival time
df['arrival_time'].replace(['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'], [0, 1, 2, 3, 4, 5], inplace=True)

#departure time
df['departure_time'].replace(['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'], [0, 1, 2, 3, 4, 5], inplace=True)

#source city
df['source_city'].replace(["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"], [0, 1, 2, 3, 4, 5], inplace= True)

#destination city
df['destination_city'].replace(["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"], [0, 1, 2, 3, 4, 5], inplace= True)

#stops
df['stops'].replace(['zero', 'one', 'two_or_more'], [0, 1, 2], inplace=True)

#%% checking for inconsistancies in the df

#Create a new function:
def num_missing(x):
  return sum(x.isnull())

#Checking for missing values in columns:
print ("Missing values per column:")
print (df.apply(num_missing, axis=0)) 

#Checking for missing values in columns:
print ("\nMissing values per row:")
print (df.apply(num_missing, axis=1).head()) 

#Checking for null values in columns:
print(f"We have {df.isnull().sum()} null values")

#Checking for duplicate values in columns:
print(f"We have {df.duplicated().sum()} duplicate values")
#
#%% subset of economy and business data 
econ=df[df['class']==0]
buz=df[df['class']==1]


# %% linear model for economy class tickets (set 1)

model_econ_1 = ols(formula='price ~ duration', data=econ)
model_econ_1_Fit = model_econ_1.fit()
print(model_econ_1_Fit.summary())

model_econ_2 = ols(formula='price ~ I(duration*duration) + (duration*stops)', data=econ)
model_econ_2_Fit = model_econ_2.fit()
print(model_econ_2_Fit.summary())

model_econ_3 = ols(formula='price ~ I(duration*duration) + (duration*stops) + days_left', data=econ)
model_econ_3_Fit = model_econ_3.fit()
print(model_econ_3_Fit.summary())



#%% linear model for business class tickets (set 1)

model_buz_1 = ols(formula='price ~ duration', data=buz)
model_buz_1_Fit = model_buz_1.fit()
print(model_buz_1_Fit.summary())

model_buz_2 = ols(formula='price ~ I(duration*duration) + (duration*stops)', data=buz)
model_buz_2_Fit = model_buz_2.fit()
print(model_buz_2_Fit.summary())

model_buz_3 = ols(formula='price ~ I(duration*duration) + (duration*stops) + days_left', data=buz)
model_buz_3_Fit = model_buz_3.fit()
print(model_buz_3_Fit.summary())
# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn import tree

# %% Regression Tree for Economy Class (set 1)
x_Air_econ=econ[['stops', 'duration', 'days_left']]
y_Air_econ=econ['price']
xtrain_econ, xtest_econ, ytrain_econ, ytest_econ = train_test_split(x_Air_econ, y_Air_econ, test_size=0.2,random_state=1)

air_tree_econ = DecisionTreeRegressor(max_depth=8, min_samples_leaf=1,random_state=50)

plane=air_tree_econ.fit(xtrain_econ, ytrain_econ)
price_pred_econ = air_tree_econ.predict(xtest_econ)

fn=['stops','duration','days_left']
tree.plot_tree(plane,feature_names=fn)

print("The mean square error is:", MSE(ytest_econ, price_pred_econ))
print("The root mean square error is:", MSE(ytest_econ, price_pred_econ)** .5)
print("The r square value is:", r2_score(ytest_econ,price_pred_econ))


MSE_CV = - cross_val_score(air_tree_econ, xtrain_econ, ytrain_econ, cv= 10, scoring='neg_mean_squared_error')
print(MSE_CV)

tree.export_graphviz(air_tree_econ,out_file="Regression_Tree_Econ_1.dot",filled = True, feature_names=fn)
#%%
number_of_observations=50
x_ax = range(len(ytest_econ[:number_of_observations]))
plt.plot(x_ax, ytest_econ[:number_of_observations], label="original")
plt.plot(x_ax, price_pred_econ[:number_of_observations], label="predicted")
plt.title("Flight Price test and predicted data")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.legend()
plt.show()

# %% Regression Tree for Business Class (set 1)
x_Air_buz=buz[['stops', 'duration', 'days_left']]
y_Air_buz=buz['price']
xtrain_buz, xtest_buz, ytrain_buz, ytest_buz = train_test_split(x_Air_buz, y_Air_buz, test_size=0.2,random_state=1)

air_tree_buz = DecisionTreeRegressor(max_depth=8, min_samples_leaf=1,random_state=50)

plane=air_tree_buz.fit(xtrain_buz, ytrain_buz)
price_pred_buz = air_tree_buz.predict(xtest_buz)

fn=['stops','duration','days_left']
tree.plot_tree(plane,feature_names=fn)

print("The mean square error is:", MSE(ytest_buz, price_pred_buz))
print("The root mean square error is:", MSE(ytest_buz, price_pred_buz)** .5)
print("The r square value is:", r2_score(ytest_buz,price_pred_buz))


MSE_CV = - cross_val_score(air_tree_buz, xtrain_buz, ytrain_buz, cv= 10, scoring='neg_mean_squared_error')
print(MSE_CV)

tree.export_graphviz(air_tree_buz,out_file="Regression_Tree_Buz.dot_1",filled = True, feature_names=fn)

# %%
number_of_observations=50
x_ax = range(len(ytest_buz[:number_of_observations]))
plt.plot(x_ax, ytest_buz[:number_of_observations], label="original")
plt.plot(x_ax, price_pred_buz[:number_of_observations], label="predicted")
plt.title("Flight Price test and predicted data")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.legend()
plt.show()
# %% knn model for economy class tickets (set 1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(xtrain_econ,ytrain_econ)
knn_price_pred_econ = knn.predict(xtest_econ)
print(knn.score(xtest_econ,ytest_econ))
print('R2 Value:',r2_score(ytest_econ, knn_price_pred_econ))

# %% knn model for business class tickets (set 1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(xtrain_buz,ytrain_buz)
knn_price_pred_econ = knn.predict(xtest_buz)
print(knn.score(xtest_buz,ytest_buz))
print('R2 Value:',r2_score(ytest_buz, knn_price_pred_econ))
# %%
# %% linear model for economy class tickets (set 2)

model_econ_1 = ols(formula='price ~ C(stops)', data=econ)
model_econ_1_Fit = model_econ_1.fit()
print(model_econ_1_Fit.summary())

model_econ_2 = ols(formula='price ~ C(stops) * days_left * C(source_city)', data=econ)
model_econ_2_Fit = model_econ_2.fit()
print(model_econ_2_Fit.summary())

model_econ_3 = ols(formula='price ~  C(stops) * days_left + duration * C(source_city) * C(destination_city)', data=econ)
model_econ_3_Fit = model_econ_3.fit()
print(model_econ_3_Fit.summary())



#%% linear model for business class tickets (set 2)

model_buz_1 = ols(formula='price ~ C(stops)', data=buz)
model_buz_1_Fit = model_buz_1.fit()
print(model_buz_1_Fit.summary())

model_buz_2 = ols(formula='price ~ C(stops) * days_left * C(source_city)', data=buz)
model_buz_2_Fit = model_buz_2.fit()
print(model_buz_2_Fit.summary())

model_buz_3 = ols(formula='price ~  C(stops) * days_left + duration * C(source_city) * C(destination_city)', data=buz)
model_buz_3_Fit = model_buz_3.fit()
print(model_buz_3_Fit.summary())

# %% Regression Tree for Economy Class (set 2)
x_Air_econ=econ[['stops', 'duration', 'days_left', 'source_city', 'destination_city']]
y_Air_econ=econ['price']
xtrain_econ, xtest_econ, ytrain_econ, ytest_econ = train_test_split(x_Air_econ, y_Air_econ, test_size=0.2,random_state=1)

air_tree_econ = DecisionTreeRegressor(max_depth=8, min_samples_leaf=1,random_state=50)

plane=air_tree_econ.fit(xtrain_econ, ytrain_econ)
price_pred_econ = air_tree_econ.predict(xtest_econ)

fn=['stops','duration','days_left','source_city','destination_city']
tree.plot_tree(plane,feature_names=fn)

print("The mean square error is:", MSE(ytest_econ, price_pred_econ))
print("The root mean square error is:", MSE(ytest_econ, price_pred_econ)** .5)
print("The r square value is:", r2_score(ytest_econ,price_pred_econ))


MSE_CV = - cross_val_score(air_tree_econ, xtrain_econ, ytrain_econ, cv= 10, scoring='neg_mean_squared_error')
print(MSE_CV)

tree.export_graphviz(air_tree_econ,out_file="Regression_Tree_Econ_2.dot",filled = True, feature_names=fn)
#%%
number_of_observations=50
x_ax = range(len(ytest_econ[:number_of_observations]))
plt.plot(x_ax, ytest_econ[:number_of_observations], label="Actual")
plt.plot(x_ax, price_pred_econ[:number_of_observations], label="Predicted")
plt.title("Actual Flight Prices vs Predicted Prices (Economy)")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.legend()
plt.show()

# %% Regression Tree for Business Class (set 2)
x_Air_buz=buz[['stops', 'duration', 'days_left', 'source_city', 'destination_city']]
y_Air_buz=buz['price']
xtrain_buz, xtest_buz, ytrain_buz, ytest_buz = train_test_split(x_Air_buz, y_Air_buz, test_size=0.2,random_state=1)

air_tree_buz = DecisionTreeRegressor(max_depth=8, min_samples_leaf=1,random_state=50)

plane=air_tree_buz.fit(xtrain_buz, ytrain_buz)
price_pred_buz = air_tree_buz.predict(xtest_buz)

fn=['stops', 'duration', 'days_left', 'source_city', 'destination_city']
tree.plot_tree(plane,feature_names=fn)

print("The mean square error is:", MSE(ytest_buz, price_pred_buz))
print("The root mean square error is:", MSE(ytest_buz, price_pred_buz)** .5)
print("The r square value is:", r2_score(ytest_buz,price_pred_buz))


MSE_CV = - cross_val_score(air_tree_buz, xtrain_buz, ytrain_buz, cv= 10, scoring='neg_mean_squared_error')
print(MSE_CV)

tree.export_graphviz(air_tree_buz,out_file="Regression_Tree_Buz2.dot",filled = True, feature_names=fn)
#%%


# %%
number_of_observations=50
x_ax = range(len(ytest_buz[:number_of_observations]))
plt.plot(x_ax, ytest_buz[:number_of_observations], label="Actual")
plt.plot(x_ax, price_pred_buz[:number_of_observations], label="Predicted")
plt.title("Actual Flight Prices vs Predicted Prices (Business)")
plt.xlabel('Observation Number')
plt.ylabel('Price')
plt.legend()
plt.show()
# %% knn model for economy class tickets (set 2)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(xtrain_econ,ytrain_econ)
knn_price_pred_econ = knn.predict(xtest_econ)
print(knn.score(xtest_econ,ytest_econ))
print('R2 Value:',r2_score(ytest_econ, knn_price_pred_econ))

# %% knn model for business class tickets (set 2)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(xtrain_buz,ytrain_buz)
knn_price_pred_econ = knn.predict(xtest_buz)
print(knn.score(xtest_buz,ytest_buz))
print('R2 Value:',r2_score(ytest_buz, knn_price_pred_econ))


#%%

# %%
