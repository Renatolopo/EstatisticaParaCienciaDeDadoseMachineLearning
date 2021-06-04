# %%
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm

import numpy as np 
import stats
import statistics
# %%
df  = pd.read_csv('../data/credit_data.csv')
df.dropna(inplace=True)
df.head()
# %%
X = df.iloc[:, 1:4].values
y = df['c#default'].values

# %%
naive_bayes = GaussianNB()
resultados_naive_bayes = cross_val_score(naive_bayes, X, y, cv=30)
resultados_naive_bayes
# %%
logistica = LogisticRegression()
resultados_logistica = cross_val_score(logistica, X, y, cv=30)
resultados_logistica
#%%
random_forest = RandomForestClassifier()
resultado_forest = cross_val_score(random_forest, X, y, cv=30)
resultado_forest

# %%
# Média
resultados_naive_bayes.mean(), resultados_logistica.mean(), resultado_forest.mean()

# %%
# Desvio padrão
resultados_naive_bayes.std(), resultados_logistica.std(), resultado_forest.std()
