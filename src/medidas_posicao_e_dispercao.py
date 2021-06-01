
# %%
import pandas as pd
# %%
df = pd.read_csv('../data/census.csv')
# %%
#media aritimetica
print('Media aritimetica:', df['age'].mean())

# %%
# media harmonica
from scipy.stats.mstats import hmean
age = [df['age'].iloc[x] for x in range(len(df))]
print('Média harmônica:',hmean(age))

# %%
# média geometrica
from scipy.stats.mstats import gmean
print('Media Geometrica:',gmean(age))

# %%
# media quadratica
import math
def quadratic_mean(dados):
    return math.sqrt(sum(n * n for n in dados) / len(dados))

print('Media quadrática:', quadratic_mean(age))
# %%
print('Mediana:', df['age'].median())
print('Moda:',df['age'].mode())