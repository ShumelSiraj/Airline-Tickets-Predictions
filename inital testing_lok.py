#%% importing in libraries
from pickle import TRUE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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
                 data=df, palette="rocket")
# %%
sns.stripplot(data=df, x='class',y='price', dodge='true',hue='airline', jitter=.5, palette='rocket')
plt.show()
#%%
df= pd.read_csv("Clean_Dataset.csv")
del df['Unnamed: 0']
#%%
df_source= pd.read_csv("Clean_Dataset.csv")
del df_source['Unnamed: 0']
coming_up=[]
for value in df_source["days_left"]:
    if 0 <= value <= 7:
        coming_up.append("Very Soon")
    if 8 <= value <= 14:
        coming_up.append("Soon")
    if 15 <= value <= 35:
        coming_up.append("Far Away")
    if value <= 36:
        coming_up.append("Very Far Away")
    
df_source['Coming_up'] = pd.Series(coming_up)   
print(df_source)
#%%
df['class'].replace(['Economy', 'Business'], [0, 1], inplace=True)

df['stops'].replace(['zero', 'one', 'two_or_more'], [0, 1, 2], inplace=True)

econ=df[df['class']==0]
econ_source=df_source[df_source['class']=='Economy']

buz=df[df['class']==1]
buz_source=df_source[df_source['class']=='Business']


#%%
#Economy
sns.set(font_scale=4)
fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
plt.xlim(0, 55)
fig1.suptitle("Scatterplot for Economy Class", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=econ[econ['stops']==0],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_ylim(1, 40000)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=econ[econ['stops']==1],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax2.set_ylim(1, 40000)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=econ[econ['stops']==2],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_ylim(1, 40000)
#%%
# Business 
sns.set(font_scale=4)
fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(55,20))
plt.xlim(0, 55)
fig1.suptitle("Scatterplot for Business Class", fontsize=60)
ax1=sns.regplot( ax=ax1, x="duration", y="price", data=buz[buz['stops']==0],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax1.set_title("0 stops", fontsize=30)
ax1.set_ylim(1, 40000)
ax2=sns.regplot( ax=ax2, x="duration", y="price", data=buz[buz['stops']==1],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax2.set_title("1 stop", fontsize=30)
ax2.set_ylim(1, 40000)
ax3=sns.regplot( ax=ax3, x="duration", y="price", data=buz[buz['stops']==2],scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax3.set_title("2 or more stops", fontsize=30)
ax3.set_ylim(1, 40000)
#%%
df_dummy= df.copy(deep=True)
# %%
df_dummy.days_left = df_dummy.days_left.astype('category')
# %%
#days_left(categorical data) vs price 
sns.barplot(x = 'Coming_up',
            y = 'price',
            data = df_source)
plt.show()
# %%
#days_left(numerical data) vs price
sns.barplot(x = 'days_left',
            y = 'price',
            data = df)
plt.show()
# %%
#Comparing price distribution for different airlines
palette = sns.color_palette("rocket")
sns.catplot(y = "price", x = "airline", data = df, kind="boxen", height = 6, aspect = 3)
plt.title("Price for Airlines",fontsize=30)
plt.xlabel("Airline", fontsize = 30)
plt.ylabel("price", fontsize = 30)
plt.show()
#%%
sns.boxplot(data=df, x='class',y='price', dodge='true',hue='airline', palette='rocket')
plt.show()
# %%
palette = sns.color_palette("rocket")
sns.catplot(y = "price", x = "class", data = df, hue='airline',kind="boxen", height = 6, aspect = 3)
plt.title("Price based on Airlines and Class",fontsize=30)
plt.xlabel("class", fontsize = 30)
plt.ylabel("price", fontsize = 30)
plt.show()
# %%
# Models
x = df[['source_city','departure_time','arrival_time','days_left','destination_city','stops','duration']].values
y = df['price'].values
class_le = LabelEncoder()
y = class_le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.3, random_state=1)
#%%
#Decision tree
# Fit dt to the training set
rf1 = DecisionTreeClassifier(criterion='entropy',random_state=0)
# Fit dt to the training set
rf1.fit(x_train,y_train)
# y_train_pred = rf1.predict(x_train)
y_test_pred = rf1.predict(x_test)
y_pred_score = rf1.predict_proba(x_test)

print('Decision Tree results')

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred)*100)

# Tree depth & leafs
print ('Tree Depth:', rf1.get_depth())
print ('Tree Leaves:', rf1.get_n_leaves())
#%%
from sklearn.linear_model import Lasso

rf2 = Lasso(random_state=0)
# Fit dt to the training set
rf2.fit(x_train,y_train)
y_test_pred = rf2.predict(x_test)
print("coefficient of determination: ",r2_score(y_test, y_test_pred))
