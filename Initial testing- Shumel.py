
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
#%%
#Pre Processing:
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
print("giving proper heading name to columns world 1")
df = df.rename(columns={'airline': 'Airline', 'flight': 'Flight','source_city':'Source_City','departure_time' : 'Departure_Time','stops':'Stops','arrival_time':'Arrival_Time','destination_city':'Destination_City', 'class': 'Class', 'duration': 'Duration','days_left':'Days_Left','price' : 'Price'})
df
#%%
df.to_csv("Final_Dataset.csv")
#%%
df= pd.read_csv("Final_Dataset.csv")
del df['Unnamed: 0']
#%%
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
#
#%%
# Frequency histogram of all the variables:
df.hist(layout=(5, 4), color='blue', figsize= (15, 12), grid=True)
plt.suptitle("Histogram plots for all variables")
#%%
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
economydf = df[(df['Class'] == 0)]
#%%
# SCattrplot of price in respect to airline economy and business class
sns.scatterplot('Airline', 'Price', data=economydf)
sns.scatterplot('Airline', 'Price', data=businessdf)
plt.legend(labels=["Economy Class","Business Class"])
plt.show()
#%%
sns.relplot(x="Duration", y="Price", hue="Stops", sizes=(15, 200), data=businessdf);
plt.title("Business Class Price With Respect To Duration And Stops.")  

sns.relplot(x="Duration", y="Price", hue="Stops", sizes=(15, 200), data=economydf);
plt.title("Business Class Price with Respect to Duration and Stops.")  

#%%
df.Airline.value_counts()

vistara=df[df['Airline']=='Vistara']
spicejet=df[df['Airline']=='SpiceJet']
indigo=df[df['Airline']=='Indigo']
air_india=df[df['Airline']=="Air_India"]
gofirst=df[df['Airline']=='GO_FIRST']
airasia=df[df['Airline']=='AirAsia']

#%%
from statsmodels.formula.api import ols
modelprice = ols(formula='Price ~ + Duration + Days_Left + Stops + Class', data=df)
print( type(modelprice) )

modelpricefit = modelprice.fit()
print( type(modelpricefit) )
print( modelpricefit.summary() )

print(f' R-squared value of the model : {modelpricefit.rsquared}')


#%%
x=df.drop(['Price', "Airline", "Flight"], axis=1)
y=df['Price']

n = 5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=n) 
knn.fit(x,y)

print(knn.score(x,y))
#%%  
