
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
#%%

df= pd.read_csv("Final_Dataset.csv")

#%%
#Pre Processing:
#
#checking for any missing values in the df
#Create a new function:
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print ("Missing values per column:")
print (df.apply(num_missing, axis=0)) 
#axis=0 defines that function is to be applied on each column

#Applying per row:
print ("\nMissing values per row:")
print (df.apply(num_missing, axis=1).head()) 
#axis=1 defines that function is to be applied on each row

print("\n")

print('Check null values')
print(df.isnull().sum())

print("\n")

print('Check for duplicate values')
print(f"We have {df.duplicated().sum()} duplicate values")
#%%
coming_up=[]
for value in df["Days_Left"]:
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
df['Coming_up'] = pd.Series(coming_up)   
print(df)

#%%
df.hist(layout=(5, 4), color='blue', figsize= (15, 12), grid=True)
plt.suptitle("Histogram plots for all variables")
#%%

# %%
print('The Airline Distribution')
print(df['Airline'].describe())
#%%
print('Airline')
plt.figure(figsize=(10,10))
sns.histplot(df['Airline'], alpha=0.45, color='red')
plt.title(" Airline")
plt.xlabel('type of Airline')
plt.ylabel("Frequency of Airline")
plt.grid()
plt.show()

#%%
df.Airline.value_counts()

vistara=df[df['Airline']=='Vistara']
spicejet=df[df['Airline']=='SpiceJet']
indigo=df[df['Airline']=='Indigo']
air_india=df[df['Airline']=="Air_India"]
gofirst=df[df['Airline']=='GO_FIRST']
airasia=df[df['Airline']=='AirAsia']

# %%
#Checking the normality
#Shapiro-Wilk test: This test is most popular to test the normality. It has below hypothesis:
#H0= The sample comes from a normal distribution.
#HA=The sample is not coming from a normal distribution.
columns= ['Source_City', 'Departure_Time', 'Stops', 'Arrival_Time', 'Destination_City', 'Class', 'Duration', 'Days_Left', 'Price']
alpha = 0.5
for i in columns:
  print([i])
  a,b= stats.shapiro(df[[i]])
  print("statistics", a, "p-value", b)
  if b < alpha:
    print("The null hypothesis can be rejected")
  else:
    print("The null hypothesis cannot be rejected")


# %%
#QQplot
import pylab as py
columns= ['Source_City', 'Departure_Time', 'Stops', 'Arrival_Time', 'Destination_City', 'Class', 'Duration', 'Days_Left', 'Price']

for i in columns:
  print([i])
  qqplot= sm.qqplot(df[[i]], line ='45')
  py.show()
  
#%%
# Because the length of the data for the economy and business is the same, we will divide the data and analyze it separately.
businessdf = df[(df['Class'] == 1)]
del businessdf['Unnamed: 0']

economydf = df[(df['Class'] == 0)]
del economydf['Unnamed: 0']
#%%
s0d3 = economydf[(economydf['Source_City'] == 0) & (economydf['Destination_City'] == 3)]
s0d3
#%%
vistara=df[df['Airline']=='Vistara']
sns.relplot(x="Duration", y="Price", hue="Stops", sizes=(15, 200), style="Class", data=vistara);
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
# %%
from statsmodels.formula.api import ols

model_vistara = ols(formula='Price ~ duration + stops + Coming_up', data=df)
model_vistara_Fit = model_vistara.fit()
print( model_vistara_Fit.summary())

# %%
%pip install lazypredict
#%%
import seaborn as sns
sns.set()
sns.catplot(y = "Price", x = "Airline", data = df.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()

#%%
x=df.drop('Price',axis=1)
y=df['Price']
#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#%%
import sklearn
estimators = sklearn.utils.all_estimators(type_filter=None)
for name, class_ in estimators:
    if hasattr(class_, 'predict_proba'):
        print(name)
        
#%%
import lazypredict
from lazypredict.Supervised import LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)
models.head(10)


'''As we can see LazyPredict gives us results of multiple models on multiple performance matrices. In the above figure, we have shown the top ten models.

Here ‘XGBRegressor’ and ‘ExtraTreesRegressor’ outperform other models significantly. It does take a high amount of training time with respect to other models. At this step we can choose priority either we want ‘time’ or ‘performance’.

We have decided to choose ‘performance’ over training time. So we will train ‘XGBRegressor and visualize the final results.'''
#%%
from xgboost import XGBRegressor
model =  XGBRegressor()
model.fit(x_train,y_train)

#let’s check Model performance…

y_pred =  model.predict(x_test)
print('Training Score :',model.score(x_train, y_train))
print('Test Score     :',model.score(x_test, y_test))

#As we can see the model score is pretty good. Let’s visualize the results of few predictions.

number_of_observations=50

x_ax = range(len(y_test[:number_of_observations]))

plt.plot(x_ax, y_test[:number_of_observations], label="original")

plt.plot(x_ax, y_pred[:number_of_observations], label="predicted")

plt.title("Flight Price test and predicted data")

plt.xlabel('Observation Number')

plt.ylabel('Price')

plt.legend()

plt.show()
# %%
