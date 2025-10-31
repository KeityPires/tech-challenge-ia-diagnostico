from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name="Modelo"):
    """
    Avalia o desempenho de um modelo de classificação e exibe as métricas principais.
    
    Parâmetros:
    -----------
    model : objeto
        Modelo treinado (deve ter o método predict()).
    X_test : array-like
        Conjunto de features de teste.
    y_test : array-like
        Rótulos reais do conjunto de teste.
    model_name : str
        Nome do modelo (para exibir nos gráficos e títulos).
    """
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas principais
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Exibir métricas numéricas
    print(f"\n Avaliação do {model_name}")
    print("-" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-score: {f1:.4f}\n")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.show()
