#%% importing in libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% reading in data 
df = pd.read_csv (r'C:\Users\pavan\Desktop\Git\6103 project\airline_data.csv') 
#%%
fig, axs = plt.subplots(2) 
fig.set_size_inches(10,10)

axs[0].plot(df['departure_time'],  df['price'],'o')
axs[1].plot(df['stops'],  df['price'],'o')
# %%
fig, axes = plt.subplots(1,2,figsize=(20,15))
sns.stripplot(ax=axes[0], data=df, x='class',y='price', dodge='true',hue='departure_time', jitter=.5)

sns.stripplot(ax=axes[1], data=df, x='class',y='price', dodge='true',hue='source_city', jitter=.5)

# %%
sc1=df[df['source_city']=='Dehli']
sc2=df[df['source_city']=='Mumbai']
sc3=df[df['source_city']=='Bangalore']
sc4=df[df['source_city']=='Kolkata']
sc5=df[df['source_city']=='Hyderabad']
sc6=df[df['source_city']=='Chennai']
data = [ sc1['price'], sc2['price'], sc3['price'], sc4['price'],sc5['price'],sc6['price'] ]
plt.boxplot(data)
# %%
