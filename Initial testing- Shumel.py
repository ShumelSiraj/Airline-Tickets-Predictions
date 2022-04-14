#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
#%%
cdf=pd.read_csv("Clean_Dataset.csv")
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
for value in df["days_left"]:
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
print(df['airline'].describe())
#%%
print('Airline')
plt.figure(figsize=(10,10))
sns.histplot(df['airline'], alpha=0.45, color='red')
plt.title(" Airline")
plt.xlabel('type of Airline')
plt.ylabel("Frequency of Airline")
plt.grid()
plt.show()

#%%
df.airline.value_counts()

vistara=df[df['airline']=='Vistara']
spicejet=df[df['airline']=='SpiceJet']
indigo=df[df['airline']=='Indigo']
air_india=df[df['airline']=="Air_India"]
gofirst=df[df['airline']=='GO_FIRST']
airasia=df[df['airline']=='AirAsia']

# %%
#Checking the normality
#Shapiro-Wilk test: This test is most popular to test the normality. It has below hypothesis:
#H0= The sample comes from a normal distribution.
#HA=The sample is not coming from a normal distribution.
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
import pylab as py
columns= ['source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left', 'price']

for i in columns:
  print([i])
  qqplot= sm.qqplot(df[[i]], line ='45')
  py.show()
# %%
from statsmodels.formula.api import ols

model_vistara = ols(formula='price ~ duration + stops + Coming_up', data=df)
model_vistara_Fit = model_vistara.fit()
print( model_vistara_Fit.summary())

# %%
