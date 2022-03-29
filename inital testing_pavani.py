# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv (r'C:\Users\pavan\Desktop\Git\6103 project\netflix_titles.csv')   
# %% histograms
df_movie=df[df['type']=='Movie']
df_TVshow=df[df['type']=='TV Show']

fig, axs = plt.subplots(2,figsize=(20,15))


axs[0].hist(df_movie['release_year'], bins='auto', alpha=0.5, edgecolor='black', linewidth=1)
axs[0].set_xlim([1900,2025])
axs[0].set_ylim([0,800])

axs[1].hist(df_TVshow['release_year'], bins='auto', alpha=0.5, edgecolor='black', linewidth=1)
axs[1].set_xlim([1900,2025])
axs[1].set_ylim([0,800])

# %% scatterplots

sns.stripplot( data=df, x='release_year',y='type', dodge='true',hue='rating', jitter=.5)
plt.legend(loc='upper right',bbox_to_anchor=(1.35,1))


# %%
df_thriller = df[df['listed_in'].str.contains("Thrillers")]

# %%
