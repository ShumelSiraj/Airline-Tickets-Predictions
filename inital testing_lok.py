#%% importing in libraries
from pickle import TRUE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% reading in data 
df = pd.read_csv (r'C:\Users\LOKESHCHOWDARY\Desktop\mining project\Clean_Dataset.csv') 
busniess = pd.read_csv (r'C:\Users\LOKESHCHOWDARY\Desktop\mining project\business.csv') 
economy=pd.read_csv (r'C:\Users\LOKESHCHOWDARY\Desktop\mining project\economy.csv') 
# %%
plt.hist(df['duration'])
plt.xlabel('duration')
plt.ylabel('fequency')
plt.show()
# %%
plt.hist(df['price'])
plt.xlabel('price')
plt.ylabel('fequency')
plt.show()

# %%
sns.catplot(x="departure_time", y="price", kind="box", data=df)
sns.catplot(x="class", y="price", kind="box", data=df)
plt.show()
# %%
sns.heatmap(df.corr(),annot=True)
plt.show()
#Economy class vs Business class
# %%
sns.boxplot(x="class", y="price",
                 data=df, palette="Set3")
# %%
sns.stripplot(data=df, x='class',y='price', dodge='true',hue='airline', jitter=.5, palette='rocket')
plt.show()
#%%
df= pd.read_csv("Clean_Dataset.csv")
del df['Unnamed: 0']

df_source= pd.read_csv("Clean_Dataset.csv")
del df_source['Unnamed: 0']

df['class'].replace(['Economy', 'Business'], [0, 1], inplace=True)

df['stops'].replace(['zero', 'one', 'two_or_more'], [0, 1, 2], inplace=True)

econ=df[df['class']==0]
econ_source=df_source[df_source['class']=='Economy']

buz=df[df['class']==1]
buz_source=df_source[df_source['class']=='Business']


#%% scatterplots for business class
fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
plt.xlim(0, 55)
fig1.suptitle("Scatterplot for Economy Class", fontsize=60)
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
#%%
# Business 
fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
plt.xlim(0, 55)
fig1.suptitle("sctterplot for business class", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=buz[buz['stops']==0],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_ylim(1, 40000)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=buz[buz['stops']==1],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax2.set_ylim(1, 40000)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=buz[buz['stops']==2],
scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_ylim(1, 40000)
#%%
df_dummy= df.copy(deep=True)
# %%
df_dummy.days_left = df_dummy.days_left.astype('category')
# %%
#days_left(categorical data) vs price 
sns.barplot(x = 'days_left',
            y = 'price',
            data = df_dummy)
plt.show()
# %%
#days_left(numerical data) vs price
sns.barplot(x = 'days_left',
            y = 'price',
            data = df)
plt.show()
# %%
sns.set()
sns.catplot(y = "price", x = "airline", data = df, kind="boxen", height = 6, aspect = 3)
plt.show()
# %%
# %%
