#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
cdf=pd.read_csv("Clean_Dataset.csv")
df= pd.read_csv("Final_Dataset.csv")
del df['Unnamed: 0']

#%%

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df.corr(), dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(df.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# %%[markdown]
# # According to the heatmap, we can conclude that :
# Highly correlated variables are :
# * stops and durantion
# * class and price
#
# Moderately correlated variable are :
# * stops and price
# * duration and departure_time
# * duration and destiantion_city
# * destination_city and stops
# * stops and price
# * class and duration
#
# Lightly correlated variable are :
# * duration and source_city
# * arrival_time and stops

#%%
df
# %%
print('airline')
plt.figure(figsize=(10,10))
sns.histplot(df['airline'], alpha=0.45, color='red')
plt.title(" airline")
plt.xlabel('type of airline')
plt.ylabel("Frequency of airline")
plt.grid()
plt.show()
#%%
print('source_city')
plt.figure(figsize=(10,10))
sns.histplot(cdf['source_city'], alpha=0.45, color='Blue')
plt.title(" source_city")
plt.xlabel('type of source_city')
plt.ylabel("Frequency of source_city")
plt.grid()
plt.show()

#%%
print('source_city')
plt.figure(figsize=(10,10))
sns.histplot(cdf['source_city'], alpha=0.45, color='Blue')
plt.title(" source_city")
plt.xlabel('type of source_city')
plt.ylabel("Frequency of source_city")
plt.grid()
plt.show()

