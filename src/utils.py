import pandas as pd

def verificar_overfitting(model, X_train, y_train, X_test, y_test, model_name="Modelo"):
    """
    Verifica possível overfitting comparando as acurácias de treino e teste.
    
    Parâmetros
    ----------
    model : objeto
        Modelo treinado (deve ter o método .score()).
    X_train, y_train : array-like
        Conjunto de dados de treinamento.
    X_test, y_test : array-like
        Conjunto de dados de teste.
    model_name : str, opcional
        Nome do modelo (para exibir nos prints).
    """
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    diff = train_acc - test_acc

    print(f"\nVerificação de Overfitting - {model_name}")
    print("-" * 45)
    print(f"Acurácia no treino: {train_acc:.4f}")
    print(f"Acurácia no teste:  {test_acc:.4f}")

    if diff > 0.05:
        print("Diferença significativa detectada — possível overfitting.")
    elif diff < -0.02:
        print("Acurácia de teste maior que a de treino — pode haver underfitting.")
    else:
        print("Modelo apresenta bom equilíbrio entre treino e teste.")


def testar_modelo(model, scaler, casos, nomes=None):
    """
    Recebe um dicionário (ou lista de dicionários) com novos casos,
    padroniza os dados e faz previsões usando o modelo treinado.

    Parâmetros
    ----------
    model : objeto
        Modelo treinado (ex: knn, árvore, etc.)
    scaler : objeto
        Scaler usado para normalizar os dados originais.
    casos : dict ou list
        Dicionário único ou lista de dicionários com os valores das variáveis.
    nomes : list, opcional
        Lista de nomes (ex: ['Paciente 1', 'Paciente 2']) para exibir junto das previsões.

    Retorna
    -------
    DataFrame com os resultados.
    """
    if isinstance(casos, dict):
        casos = [casos]  # converte um único caso em lista

    df_novos = pd.DataFrame(casos)

    # Padronizar com o mesmo scaler usado no treino
    X_novos = scaler.transform(df_novos)

    # Fazer previsões
    preds = model.predict(X_novos)

    # Criar DataFrame de resultados
    resultados = pd.DataFrame(df_novos)
    resultados['Previsão'] = ['MALIGNO (1)' if p == 1 else 'BENIGNO (0)' for p in preds]

    # Adicionar nomes, se houver
    if nomes:
        resultados.insert(0, 'Paciente', nomes)

    print("\n🔎 Resultados das previsões:\n")
    display(resultados[['Previsão'] + ([col for col in resultados.columns if col not in ['Previsão', 'Paciente']] or [])])

    return resultados