#%%
import numpy as np
import pandas as pd
from prometheus_client import Summary
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
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
corr = df.corr()
ax1 = sns.heatmap(corr, cbar=0, linewidths=2,vmax=1, vmin=0, square=True, cmap='Blues', annot=True,annot_kws = {'size':5})
plt.title("correlation of all the variables")
plt.show()
# %%
df

#%%stripplots to look at economy and business
fig, axes = plt.subplots(1,3,figsize=(30,20))

sns.stripplot(ax=axes[0], data=df, x='class',y='price', dodge='true',hue='stops', jitter=.5, palette='rocket')

sns.stripplot(ax=axes[1], data=df, x='class',y='price', dodge='true',hue='arrival_time', jitter=.5,palette='rocket')

sns.stripplot(ax=axes[2], data=df, x='class',y='price', dodge='true',hue='departure_time', jitter=.5,palette='rocket')


# %% scatterplots for each airline for different number of stops

# Air_India
fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20), sharey=True)
air_india=df[df['airline']=="Air_India"]
fig1.suptitle("Airline: Air_India", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=air_india[air_india['stops']==0],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_xlim(0,55)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=air_india[air_india['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax2.set_xlim(0,55)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=air_india[air_india['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_xlim(0,55)

# SpiceJet
fig2, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20),sharey=True)
spicejet=df[df['airline']=='SpiceJet']
fig2.suptitle("Airline: SpiceJet", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=spicejet[spicejet['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_xlim(0,55)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=spicejet[spicejet['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax1.set_xlim(0,55)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=spicejet[spicejet['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_xlim(0,55)

# Indigo
fig3, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20),sharey=True)
indigo=df[df['airline']=='Indigo']
fig3.suptitle("Airline: Indigo", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=indigo[indigo['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_xlim(0,55)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=indigo[indigo['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax1.set_xlim(0,55)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=indigo[indigo['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax1.set_xlim(0,55)

# GO_FIRST
fig4, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20),sharey=True)
gofirst=df[df['airline']=='GO_FIRST']
fig4.suptitle("Airline: GO_FIRST", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=gofirst[gofirst['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_xlim(0,55)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=gofirst[gofirst['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax2.set_xlim(0,55)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=gofirst[gofirst['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_xlim(0,55)

# Vistara
fig5, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20),sharey=True)
vistara=df[df['airline']=='Vistara']
fig5.suptitle("Airline: Vistara", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=vistara[vistara['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_xlim(0,55)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=vistara[vistara['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax2.set_xlim(0,55)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=vistara[vistara['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_xlim(0,55)

# AirAsia
fig6, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20),sharey=True)
airasia=df[df['airline']=='AirAsia']
fig6.suptitle("Airline: AirAsia", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=airasia[airasia['stops']==0],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_xlim(0,55)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=airasia[airasia['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax2.set_xlim(0,55)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=airasia[airasia['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_xlim(0,55)
#%% linear regression AirAsia
model_AirAsia_1 = ols(formula='price ~ duration', data=airasia)
model_AirAsia_1_Fit = model_AirAsia_1.fit()
print( model_AirAsia_1_Fit.summary())

model_AirAsia_2 = ols(formula='price ~ duration + stops', data=airasia)
model_AirAsia_2_Fit = model_AirAsia_2.fit()
print( model_AirAsia_2_Fit.summary())

model_AirAsia_3 = ols(formula='price ~ duration + stops + days_left', data=airasia)
model_AirAsia_3_Fit = model_AirAsia_3.fit()
print( model_AirAsia_3_Fit.summary())
# %%
