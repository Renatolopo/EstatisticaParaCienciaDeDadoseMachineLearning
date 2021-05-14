# %%
import pandas as pd
import numpy as np 
import random
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE

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

# undersampling
tl = TomekLinks(return_indices=True, ratio='majority')
x_under, y_under, id_under = tl.fit_sample(x, y)
clf_under = RandomForestClassifier()
clf_under.fit(x_under, y_under)

# oversampling
smote = SMOTE(ration='minority')
x_over, y_over = smote.fit(x, y)
clf_over = RandomForestClassifier()
clf_over.fit(x_over, y_over)

# %%
predict_under = clf_under.predict(x_teste)
predict_over = clf_over.predict(x_teste)

# %%
accuracy_score(predict_under, y_teste)
accuracy_score(predict_under, y_teste)
# %%