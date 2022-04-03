#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
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
df
# %%
corr = df.corr()
ax1 = sns.heatmap(corr, cbar=0, linewidths=2,vmax=1, vmin=0, square=True, cmap='Blues')
plt.title("correlation of all the variables")
plt.show()
# %%
fig, axes = plt.subplots(1,3,figsize=(20,20))

sns.stripplot(ax=axes[0], data=df, x='class',y='price', dodge='true',hue='stops', jitter=.5, palette='rocket')

sns.stripplot(ax=axes[1], data=df, x='class',y='price', dodge='true',hue='arrival_time', jitter=.5,palette='rocket')

sns.stripplot(ax=axes[2], data=df, x='class',y='price', dodge='true',hue='departure_time', jitter=.5,palette='rocket')

# %%
sns.boxplot(x="stops", y="price", data=df,palette='rocket')
#%%
sc1=df[df['stops']==0]
sc2=df[df['stops']==1]
sc3=df[df['stops']==2]

sns.lmplot(x="duration", y="price", data=sc3,palette='rocket', hue="airline")




# %%
spicejet=df[df['airline']=='SpiceJet']
sc_Indigo=df[df['airline']=="Indigo"]
sc_GO_FIRST=df[df['airline']=="GO_FIRST"]
sc3=df[df['airline']=="Vistara"]
sc3=df[df['airline']=="AirAsia"]
sc3=df[df['airline']=="Air_India"]

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

sns.lmplot( x="duration", y="price", data=sc_Indigo[sc_Indigo['stops']==0] ,palette='rocket', hue="stops", ax=ax1)



# %%
spicejet=df[df['airline']=='SpiceJet']
spicejet=spicejet.pivot_table(index='source_city', columns ='destination_city', values='price', aggfunc='mean')
sns.heatmap(spicejet)

# %%

# %%
spicejet=df[df['airline']=='SpiceJet']
spicejet=spicejet[['price','duration','stops']]
sns.pairplot(spicejet, hue ='stops')

# %%
