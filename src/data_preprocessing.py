import pandas as pd
from sklearn.preprocessing import StandardScaler

def carregar_dados(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

def separar_features_target(df):
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    return X, y

def padronizar_dados(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
