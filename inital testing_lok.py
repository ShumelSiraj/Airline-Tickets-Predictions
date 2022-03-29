#%% importing in libraries
from pickle import TRUE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% reading in data 
df = pd.read_csv (r'C:\Users\LOKESHCHOWDARY\Desktop\mining project\Clean_Dataset.csv') 

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
sns.stripplot(x="class", y="price", data=df,hue='airline')
# %%
sns.heatmap(df.corr(),annot=True)
plt.show()
# %%
