# %%
import pandas as pd 
import numpy as np 

# %%
df = pd.read_csv('../data/census.csv')
df
# %%
df['age'].plot.hist()
# %%
df['age'] = pd.cut(df['age'], bins=[0, 17, 25, 40, 60, 90],
labels=['Faixa1', 'Faixa2', 'Faixa3','Faixa4', 'Faixa5'])

# %%
df.head()

# %%
df['age'].unique()
