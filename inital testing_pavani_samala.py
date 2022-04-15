#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
# %%
df_source= pd.read_csv("Clean_Dataset.csv")
del df_source['Unnamed: 0']

coming_up=[]
for value in df_source["days_left"]:
    if 0 <= value <= 7:
        coming_up.append("Very Soon")
    if 8 <= value <= 14:
        coming_up.append("Soon")
    if 15 <= value <= 35:
        coming_up.append("Far Away")
    if value <= 36:
        coming_up.append("Very Far Away")
    else:
        coming_up.append("NA")   
df_source['Coming_up'] = pd.Series(coming_up)   
print(df_source)


#%%
df= pd.read_csv("Clean_Dataset.csv")
del df['Unnamed: 0']

#%%[markdown]
# Changed the "class" column values to numeric :   <br>
#
# "Economy" = 0   <br>
# "Business" = 1

# %%
df['class'].replace(['Economy', 'Business'], [0, 1], inplace=True)

#%%[markdown]
# Changed the "arrival_time" and "departure_time" column values to numeric :<br>
#
# 'Early_Morning' = 0   <br>
# 'Morning' = 1         <br>
# 'Afternoon' = 2       <br>
# 'Evening' = 3         <br>
# 'Night' = 4           

# %%
df['arrival_time'].replace(['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'], [0, 1, 2, 3, 4, 5], inplace=True)
df['departure_time'].replace(['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'], [0, 1, 2, 3, 4, 5], inplace=True)

#%%[markdown]
# Changed the "source_city" and "destination_city" column values to numeric :  <br>    
#
# 'Mumbai' = 0     <br>
# 'Delhi' = 1      <br>
# 'Bangalore' = 2  <br> 
# 'Kolkata' = 3    <br> 
# 'Hyderabad' = 4  <br>  
# 'Chennai' = 5    <br>  

#%%
df['source_city'].replace(["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"], [0, 1, 2, 3, 4, 5], inplace= True)
df['destination_city'].replace(["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"], [0, 1, 2, 3, 4, 5], inplace= True)
#%%[markdown]
# Changed the "stops" column values to numeric : <br>
#
# "zero" = 0   <br>
# "one" = 1    <br>
# "two_or_more" = 2

#%%
df['stops'].replace(['zero', 'one', 'two_or_more'], [0, 1, 2], inplace=True)
# %%
corr = df.corr()
ax1 = sns.heatmap(corr, cbar=0, linewidths=2,vmax=1, vmin=0, square=True, cmap='Blues', annot=True,annot_kws = {'size':5})
plt.title("correlation of all the variables")
plt.show()
# %%
df

#%% subset of economy and business data 
econ=df[df['class']==0]
econ_source=df_source[df_source['class']=='Economy']

buz=df[df['class']==0]
buz_source=df_source[df_source['class']=='Business']
# %% scatterplots for each airline for different number of stops

# Economy
fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
plt.xlim(0, 55)
fig1.suptitle("Airline: Air_India", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=econ[econ['stops']==0],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_ylim(1, 40000)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=econ[econ['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax2.set_ylim(1, 40000)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=econ[econ['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_ylim(1, 40000)

# Business


# %% linear model for economy class tickets

model_econ_1 = ols(formula='price ~ duration', data=econ)
model_econ_1_Fit = model_econ_1.fit()
print(model_econ_1_Fit.summary())

model_econ_2 = ols(formula='price ~ I(duration*duration) + (duration*stops)', data=econ_source)
model_econ_2_Fit = model_econ_2.fit()
print(model_econ_2_Fit.summary())

model_econ_3 = ols(formula='price ~ duration * stops * Coming_up', data=econ_source)
model_econ_3_Fit = model_econ_3.fit()
print(model_econ_3_Fit.summary())



#%% linear model for business class tickets

model_buz_1 = ols(formula='price ~ duration', data=buz)
model_buz_1_Fit = model_buz_1.fit()
print(model_buz_1_Fit.summary())

model_buz_2 = ols(formula='price ~ I(duration*duration) + (duration*stops)', data=buz_source)
model_buz_2_Fit = model_buz_2.fit()
print(model_buz_2_Fit.summary())

model_buz_3 = ols(formula='price ~ duration * stops * Coming_up', data=buz_source)
model_buz_3_Fit = model_buz_3.fit()
print(model_buz_3_Fit.summary())
# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn import tree

# %%
x_Air=econ[['stops', 'duration', 'days_left']]
y_Air=econ['price']
xtrain, xtest, ytrain, ytest = train_test_split(x_Air, y_Air, test_size=0.2,random_state=1)

air_tree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.1,random_state=44)

plane=air_tree.fit(xtrain, ytrain)
price_pred = air_tree.predict(xtest)

fn=['stops','duration','days_left']
tree.plot_tree(plane,feature_names=fn)

print("The mean square error is:", MSE(ytest, price_pred))
print("The root mean square error is:", MSE(ytest, price_pred)** .5)
print("The r square value is:", r2_score(ytest,price_pred))


MSE_CV = - cross_val_score(air_tree, xtrain, ytrain, cv= 10, scoring='neg_mean_squared_error')
print(MSE_CV)
# %%
plt.savefig('out.pdf')
# %%
#tree.export_graphviz(air_tree,out_file="tree.dot",filled = True)
# %%
x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, price_pred, label="predicted")
# %%
