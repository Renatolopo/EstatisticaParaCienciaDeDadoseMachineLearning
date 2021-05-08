import random
import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit

def get_amostragem_aleatoria_simples(df, amostras):
    return df.sample(n = amostras)

def get_amostragem_sistematica(df, amostras):
    intervalo = len(df) // amostras
    random.seed(2)
    inicio = random.randint(0, intervalo)
    indices = np.arange(inicio, len(df), step=intervalo)
    amostra_sistematica = df.iloc[indices]
    return amostra_sistematica

def get_amostragem_agrupamento(df, amostras):
    numero_grupos = len(df) / amostras
    intervalo = len(df)/ numero_grupos

    grupos = []
    id_grupo = 0
    contagem = 0
    for _ in df.iterrows():
        grupos.append(id_grupo)
        contagem += 1
        if contagem > intervalo:
            contagem = 0 
            id_grupo += 1

    df['grupo'] = grupos 
    random.seed(1)
    grupo_selecionado = random.randint(0, numero_grupos)
    return df[df['grupo'] == grupo_selecionado]

def get_amostragem_estratificada(df, amostras):
    split = StratifiedShuffleSplit(test_size=(amostras/len(df)))
    for x, y in split.split(df, df['c#default']):
        df_x = df.iloc[x]
        df_y = df.iloc[y]
    
    print(df_x.shape)
    print(df_y.shape)
    return df_y

def get_amostragem_reservatorio(df, amostras):
    stream = []
    for i in range(len(df)):
        stream.append(i)

    i = 0
    tamanho = len(df)
    reservatorio = [0] * amostras 
    for i in range(amostras):
        reservatorio[i] = stream[i]

    while i < tamanho:
        j = random.randrange(i+1)
        if j < amostras:
            reservatorio[j] = stream[i]
        i += 1
    return  df.iloc[reservatorio]


df = pd.read_csv('./data/credit_data.csv')
df_amostragem_aleatoria_simples = get_amostragem_aleatoria_simples(df, 1000)
df_amostragem_sistematica = get_amostragem_sistematica(df, 1000)
df_amostragem_agrupamento = get_amostragem_agrupamento(df, 1000)
df_amostragem_estratificada = get_amostragem_estratificada(df, 1000)
df_amostragem_reservatrio = get_amostragem_reservatorio(df, 1000)
print(df_amostragem_reservatrio) 