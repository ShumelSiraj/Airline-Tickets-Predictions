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



# %%



# %%
spicejet=df[df['airline']=='SpiceJet']
spicejet=spicejet.pivot_table(index='source_city', columns ='destination_city', values='price', aggfunc='mean')
sns.heatmap(spicejet)

# %% scatterplots for each airline for different number of stops

# Air_India
fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
plt.xlim(0, 55)
plt.ylim(0,123071)

air_india=df[df['airline']=="Air_India"]
sns.regplot( ax=ax1, x="duration", y="price", data=air_india[air_india['stops']==0],scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax2, x="duration", y="price", data=air_india[air_india['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax3, x="duration", y="price", data=air_india[air_india['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})

# SpiceJet
fig2, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
spicejet=df[df['airline']=='SpiceJet']
sns.regplot( ax=ax1, x="duration", y="price", data=spicejet[spicejet['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax2, x="duration", y="price", data=spicejet[spicejet['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax3, x="duration", y="price", data=spicejet[spicejet['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})

# Indigo
fig3, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
indigo=df[df['airline']=='Indigo']
sns.regplot( ax=ax1, x="duration", y="price", data=indigo[indigo['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax2, x="duration", y="price", data=indigo[indigo['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax3, x="duration", y="price", data=indigo[indigo['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})

# GO_FIRST
fig4, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
gofirst=df[df['airline']=='GO_FIRST']
sns.regplot( ax=ax1, x="duration", y="price", data=gofirst[gofirst['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax2, x="duration", y="price", data=gofirst[gofirst['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax3, x="duration", y="price", data=gofirst[gofirst['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})

# Vistara
fig5, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
vistara=df[df['airline']=='Vistara']
sns.regplot( ax=ax1, x="duration", y="price", data=vistara[vistara['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax2, x="duration", y="price", data=vistara[vistara['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax3, x="duration", y="price", data=vistara[vistara['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})

# AirAsia
fig6, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
airasia=df[df['airline']=='AirAsia']
sns.regplot( ax=ax1, x="duration", y="price", data=airasia[airasia['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax2, x="duration", y="price", data=airasia[airasia['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
sns.regplot( ax=ax3, x="duration", y="price", data=airasia[airasia['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})

# %%
spicejet=df[df['airline']=='SpiceJet']
spicejet=spicejet[['price','duration','stops']]
sns.pairplot(spicejet, hue ='stops')

# %%
sns.violinplot(x="stops", y="duration", data=airasia ,palette='rocket')
# %%
plt.hist(df['price'])
plt.xlabel('price')
plt.ylabel('fequency')
plt.show()
#%%