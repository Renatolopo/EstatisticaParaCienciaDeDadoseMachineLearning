import random
import numpy as np 
import pandas as pd 

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


df = pd.read_csv('./data/credit_data.csv')
df_amostragem_aleatoria_simples = get_amostragem_aleatoria_simples(df, 1000)
df_amostragem_sistematica = get_amostragem_sistematica(df, 1000)
df_amostragem_agrupamento = get_amostragem_agrupamento(df, 1000)
print(df.shape) 