
#%% importing libraries
from os import remove
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro
import statsmodels.api as sm
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
#%%
# EDA
print(df.shape)
print(df.info())
print(df.describe())
#%% correlation map
corr = df.corr()
corr.style.background_gradient(cmap='plasma')

# df_corr value > 0.1
df_corr = df.corr()['price'][:-1]
important_feature_list = df_corr[abs(df_corr) > 0.1].sort_values(ascending=False)
print("There is {} strongly correlated values greater than 0.1 with Price:\n{}".format(len(important_feature_list), important_feature_list))
#%%
#Comparing price distribution for different airlines
palette = sns.color_palette("rocket")
sns.catplot(y = "price", x = "airline", data = df, kind="boxen", height = 6, aspect = 3)
plt.title("Price for Airlines",fontsize=30)
plt.xlabel("Airline", fontsize = 30)
plt.ylabel("price", fontsize = 30)
plt.show()

#%%
#price comparision based on airlines and classes
palette = sns.color_palette("rocket")
sns.catplot(y = "price", x = "class", data = df, hue='airline',kind="boxen", height = 6, aspect = 3)
plt.title("Price based on Airlines and Class",fontsize=30)
plt.xlabel("class", fontsize = 30)
plt.ylabel("price", fontsize = 30)
plt.show()
#%% Ticket Price Distribution
plt.figure(figsize=(9, 8))
sns.distplot(df['price'], color='g', bins=100, hist_kws={'alpha': 0.4});


#%% Numerical data distribution
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


#%% THIS NEEDS TO BE FIXED
# from scipy.stats import shapiro
# data = df(columns= ['source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left', 'price'])
# stats, p = shapiro(data)
#%%
#Checking the normality

columns= ['source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left', 'price']
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
import numpy as np
import pylab
import scipy.stats as stats

#qqdf = df[['Source_City', 'Departure_Time', 'Stops', 'Arrival_Time', 'Destination_City', 'Class', 'Duration', 'Days_Left', 'Price']]
stats.probplot(df['source_city'], dist='norm', plot=pylab)
plt.title("Source City")
pylab.show()
stats.probplot(df['destination_city'], dist='norm', plot=pylab)
plt.title("Source City")
pylab.show()
stats.probplot(df['arrival_time'], dist='norm', plot=pylab)
plt.title("Arrival Time")
pylab.show()
stats.probplot(df['departure_time'], dist='norm', plot=pylab)
plt.title("Departure Time")
pylab.show()
stats.probplot(df['stops'], dist='norm', plot=pylab)
plt.title("Stops")
pylab.show()
stats.probplot(df['class'], dist='norm', plot=pylab)
plt.title("Class")
pylab.show()
stats.probplot(df['duration'], dist='norm', plot=pylab)
plt.title("Duration")
pylab.show()
stats.probplot(df['days_left'], dist='norm', plot=pylab)
plt.title("Days left")
pylab.show()
stats.probplot(df['price'], dist='norm', plot=pylab)
plt.title("Price")
pylab.show()
  
#%% subset based on class
econ=df[df['class']==0]
buz=df[df['class']==1]
#%%
# Scatterplot

#%%
