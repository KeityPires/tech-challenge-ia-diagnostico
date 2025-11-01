from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def train_knn_model(X_train, y_train, n_neighbors=5):
    """
    Treina um modelo KNN com os dados fornecidos.    
    Parâmetros:
    X_train : DataFrame ou array
        Conjunto de dados de treinamento
    y_train : Series ou array
        Rótulos (diagnóstico)
    n_neighbors : int, opcional (default=5)
        Número de vizinhos considerados pelo KNN.    
    Retorna:
    knn : objeto KNeighborsClassifier treinado
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    print("KNN treinado com sucesso.")
    return knn

def train_decision_tree_model(X_train, y_train, random_state=42, **kwargs):
    """
    Treina um modelo de Árvore de Decisão com os dados fornecidos.
    Parâmetros:
    X_train : DataFrame ou array
        Conjunto de dados de treinamento (features)
    y_train : Series ou array
        Rótulos (diagnóstico)
    random_state : int, opcional (default=42)
        Define a semente para reprodutibilidade dos resultados.
    **kwargs : parâmetros adicionais
        Hiperparâmetros do DecisionTreeClassifier, ex: max_depth, criterion, etc.
    Retorna:
    model : objeto DecisionTreeClassifier treinado
    """
    model = DecisionTreeClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    print("Árvore de Decisão treinada com sucesso.")
    return model


