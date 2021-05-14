# %%
import pandas as pd
import numpy as np 
import random
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# Objetivo: Prever a confiança de usuários baseado em 
# traços de personalidade extraídos de textos

# %%
df = pd.read_csv('../data/csv_result-ebay_confianca_completo.csv')
df = df.dropna()
reputacao = [(1 if df['reputation'].iloc[x] == 'Bom' else 0) for x in range(len(df))]
df['reputacao'] = reputacao
df_dummies = pd.get_dummies((df[['blacklist']]))
df = pd.concat([df,df_dummies], axis=1)
df = df.drop(columns=['blacklist', 'reputation'])
df
#%%
x = df.drop(columns=['reputacao'])
y = df['reputacao']
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.2, stratify=y)
# %%

sns.countplot(df['reputacao'])
# %%
model = GaussianNB()
model.fit(x_treinamento, y_treinamento)
# %%
predict = model.predict(x_teste)
predict
# %%
accuracy_score(predict, y_teste)
# %%
