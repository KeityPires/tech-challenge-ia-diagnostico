import sys, os
import unittest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import carregar_dados, separar_features_target, padronizar_dados
from src.model_training import train_knn_model, train_decision_tree_model


DATA_PATH = "data/data.csv"


class TestPipeline(unittest.TestCase):
    """Testes básicos do pipeline de Machine Learning"""

    def test_carregar_dados(self):
        """Verifica se o dataset é carregado corretamente"""
        df = carregar_dados(DATA_PATH)
        self.assertFalse(df.empty, "O dataframe não deve estar vazio.")
        self.assertIn("diagnosis", df.columns, "A coluna 'diagnosis' deve estar presente no dataset.")

    def test_divisao_dados(self):
        """Verifica se a separação entre features e target está correta"""
        df = carregar_dados(DATA_PATH)
        X, y = separar_features_target(df)
        self.assertEqual(len(X), len(y), "As features e o target devem ter o mesmo número de amostras.")
        self.assertNotIn("diagnosis", X.columns, "A coluna 'diagnosis' não deve estar nas features.")

    def test_treinamento_modelos(self):
        """Verifica se os modelos KNN e Decision Tree são treinados corretamente"""
        df = carregar_dados(DATA_PATH)
        X, y = separar_features_target(df)
        X_train_scaled, X_test_scaled, scaler = padronizar_dados(X, X)

        knn = train_knn_model(X_train_scaled, y)
        tree = train_decision_tree_model(X_train_scaled, y)

        self.assertTrue(hasattr(knn, "predict"), "O modelo KNN não foi treinado corretamente.")
        self.assertTrue(hasattr(tree, "predict"), "O modelo Decision Tree não foi treinado corretamente.")


if __name__ == "__main__":
    unittest.main()
