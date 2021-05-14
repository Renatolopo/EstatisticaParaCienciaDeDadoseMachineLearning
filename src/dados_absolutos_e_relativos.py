#%%
import pandas as pd 
#%%
df = pd.read_csv('../data/census.csv')
df = df[['education', 'income']]

df
# %%
df2 = df.groupby(['education','income'])['education'].count()
