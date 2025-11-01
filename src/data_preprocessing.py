import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados(caminho_arquivo):
    """
    Carrega o dataset CSV, remove colunas irrelevantes e codifica o diagnóstico.
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
        print(f"Dataset carregado com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    except FileNotFoundError:
        raise FileNotFoundError("Arquivo não encontrado. Verifique o caminho informado.")
    
    # Converter diagnóstico para 1 (maligno) e 0 (benigno)
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])
    else:
        raise ValueError("Coluna 'diagnosis' não encontrada no dataset.")
    return df


def separar_features_target(df, target='diagnosis'):
    """
    Separa as variáveis independentes (X) e dependente (y).
    """
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def dividir_dados(X, y, test_size=0.2, random_state=42):
    """
    Divide os dados em treino e teste.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def padronizar_dados(X_train, X_test):
    """
    Aplica padronização (normalização) nas features numéricas.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def analisar_correlacao(df, target='diagnosis', top_n=10):
    """
    Calcula a correlação entre as variáveis e o target e plota a matriz
    das variáveis mais correlacionadas.
    Parâmetros:
    df : pandas.DataFrame
        DataFrame com os dados já tratados (sem colunas não numéricas).
    target : str, opcional
        Nome da variável alvo (default='diagnosis').
    top_n : int, opcional
        Número de variáveis mais correlacionadas a exibir (default=10).
    Retorna:    
    corr_with_target : pandas.Series
        Correlação de cada variável com o target.
    """
    # Calcula matriz de correlação
    corr_matrix = df.corr().round(2)

    # Ordena pela correlação com o target
    corr_with_target = corr_matrix[target].sort_values(ascending=False)

    print(f"Correlação das variáveis com o target '{target}':")
    display(corr_with_target.head(top_n + 1))  # +1 para incluir o próprio target

    # Seleciona top N variáveis (excluindo a própria target)
    top_features = corr_with_target.index[1:top_n + 1]

    # Plota matriz de correlação das top variáveis
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Matriz de Correlação - Top {top_n + 1} Variáveis Mais Relevantes', fontsize=12)
    plt.show()

    return corr_with_target