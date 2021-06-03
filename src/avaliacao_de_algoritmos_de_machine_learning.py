# %%
import pandas as pd 
df  = pd.read_csv('../data/credit_data.csv')
# %%
df.dropna(inplace=True)
df.shape
# %%
df.head()
# %%
X = df.iloc[:, 1:4].values
X
# %%
y = df['c#default'].values
y
# %%
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# %%
resultados_naive_bayes = []
resultados_logistica = []
resulotado_forest =[]
for i in range(30):
    X_treinamento, X_test, y_treinamento, y_test = train_test_split(X, y, test_size=0.2,
    stratify=y, random_state=i)

    naive_bayes = GaussianNB()
    naive_bayes.fit(X_treinamento, y_treinamento)
    resultados_naive_bayes.append(accuracy_score(y_test, naive_bayes.predict(X_test)))

    logistica = LogisticRegression()
    logistica.fit(X_treinamento, y_treinamento)
    resultados_logistica.append(accuracy_score(y_test, logistica.predict(X_test)))

    random_forest = RandomForestClassifier()
    random_forest.fit(X_treinamento, y_treinamento)
    resulotado_forest.append(accuracy_score(y_test, random_forest.predict(X_test)))
    
 
# %%
print(f'Naive Bayes: {resultados_naive_bayes}\n')
print(f'Regreção Logistica: {resultados_logistica}\n')
print(f'Random forest: {resulotado_forest}\n')
# %%
import numpy as np 
import stats
import statistics
# %%
resultados_naive_bayes = np.array(resultados_naive_bayes)
resultados_logistica = np.array(resultados_logistica)
resulotado_forest = np.array(resulotado_forest)
# %%
#Media
resultados_naive_bayes.mean(), resultados_logistica.mean(), resulotado_forest.mean()

# %%
#Moda
statistics.mode(resultados_naive_bayes), statistics.mode(resultados_logistica), statistics.mode(resulotado_forest)

# %%
np.median(resultados_naive_bayes), np.median(resultados_logistica), np.median(resulotado_forest)

# %%
# Variância
np.set_printoptions(suppress=True)
np.var(resultados_naive_bayes), np.var(resultados_logistica), np.var(resulotado_forest)

# %%
np.min([8.756250000000001e-05, 0.00020933333333333337, 4.320138888888897e-05])
# %%
# Desvio padrão
np.std(resultados_naive_bayes), np.std(resultados_logistica), np.std(resulotado_forest)

# %%
# Coeficiente de variação
stats.variation(resultados_naive_bayes) * 100, stats.variation(resultados_logistica) * 100, stats.variation(resulotado_forest) * 100

# %%
