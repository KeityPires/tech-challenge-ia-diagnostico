import os
import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import carregar_dados, separar_features_target, padronizar_dados

from src.genetic_optimization_tree import (
    GAConfigTree,
    optimize_tree_with_ga,
)

from src.genetic_optimization_knn import (
    GAConfig,
    optimize_knn_with_ga,
)

from src.llm_interpretation import (
    CaseExplanationInput,
    build_prompt_case,
    build_prompt_report,
    checklist_quality,
    call_gpt,
)

DATA_PATH = "data/data.csv"


class TestGeneticOptimizationTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = carregar_dados(DATA_PATH)
        X, y = separar_features_target(df)

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        cls.X_train = X_train
        cls.y_train = y_train

    def test_optimize_tree_with_ga_runs_and_returns(self):
        """Verifica se o GA da árvore roda e retorna modelo treinado + params + history."""
        config = GAConfigTree(
            population_size=6,
            n_generations=2,
            cv_folds=2,
            mut_prob=0.3,
            cx_prob=0.7,
            tournament_k=2,
            random_state=42,
        )

        model, best_params, history = optimize_tree_with_ga(self.X_train, self.y_train, config=config)

        # modelo treinado (DecisionTree tem tree_ após fit)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "tree_"), "Árvore deveria estar ajustada (fit) ao final do GA.")

        # melhores params
        self.assertIsInstance(best_params, dict)
        expected_keys = {
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "criterion",
            "splitter",
            "max_features",
            "class_weight",
        }
        self.assertTrue(expected_keys.issubset(best_params.keys()))

        # histórico
        self.assertIsInstance(history, pd.DataFrame)
        self.assertFalse(history.empty)
        self.assertIn("generation", history.columns)
        self.assertIn("fitness_max", history.columns)


class TestGeneticOptimizationKNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = carregar_dados(DATA_PATH)
        X, y = separar_features_target(df)

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # KNN precisa de escala
        X_train_scaled, _, _ = padronizar_dados(X_train, X_train)

        cls.X_train = np.asarray(X_train_scaled)
        cls.y_train = np.asarray(y_train)

    def test_optimize_knn_with_ga_runs_and_returns(self):
        """Verifica se o GA do KNN roda e retorna best_model + params + history."""
        config = GAConfig(
            population_size=8,
            n_generations=2,
            cv_folds=2,
            cx_prob=0.7,
            mut_prob=0.3,
            tournament_size=2,
            random_state=42,
            elitism=1,
            use_fn_penalty=False,  
        )

        best_model, best_params, history = optimize_knn_with_ga(self.X_train, self.y_train, config=config)

        self.assertTrue(hasattr(best_model, "predict"))
        self.assertIsInstance(best_params, dict)

        expected_keys = {"n_neighbors", "weights", "metric"}
        self.assertTrue(expected_keys.issubset(best_params.keys()))

        self.assertIsInstance(history, pd.DataFrame)
        self.assertFalse(history.empty)
        self.assertIn("generation", history.columns)
        self.assertIn("fitness_max", history.columns)

    def test_best_model_can_fit_and_predict(self):
        """Como o KNN retornado é instanciado, validamos que ele consegue fit/predict."""
        config = GAConfig(
            population_size=6,
            n_generations=1,
            cv_folds=2,
            random_state=42,
            elitism=1,
            use_fn_penalty=False,
        )

        model, params, history = optimize_knn_with_ga(self.X_train, self.y_train, config=config)

        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_train[:10])

        self.assertEqual(len(preds), 10)
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))


class TestLLMInterpretation(unittest.TestCase):
    def test_build_prompt_case_contains_required_sections(self):
        case = CaseExplanationInput(
            model_name="DecisionTree_GA",
            pred_label=1,
            p_maligno=0.87,
            top_features={"radius_mean": 0.81234, "concavity_mean": 0.77111},
            clinical_notes="",
        )

        prompt = build_prompt_case(case)

        self.assertIn("assistente clínico", prompt.lower())
        self.assertIn("Modelo: DecisionTree_GA", prompt)
        self.assertIn("0=benigno, 1=maligno", prompt)
        self.assertIn("Não substitui avaliação médica", prompt)
        self.assertIn("radius_mean=0.812", prompt)  
        self.assertIn("concavity_mean=0.771", prompt)

    def test_build_prompt_case_without_features(self):
        case = CaseExplanationInput(
            model_name="KNN_GA",
            pred_label=0,
            p_maligno=0.10,
            top_features=None,
            clinical_notes="",
        )
        prompt = build_prompt_case(case)
        self.assertIn("não disponível", prompt.lower())

    def test_build_prompt_report_contains_metrics(self):
        metrics = {
            "model_name": "KNN_GA",
            "acc": 0.95,
            "rec": 0.92,
            "f1": 0.93,
            "tn": 70,
            "fp": 2,
            "fn": 3,
            "tp": 39,
        }

        prompt = build_prompt_report(metrics)
        self.assertIn("Modelo: KNN_GA", prompt)
        self.assertIn("Accuracy: 0.950", prompt)
        self.assertIn("Recall (maligno=1): 0.920", prompt)
        self.assertIn("F1 (maligno=1): 0.930", prompt)
        self.assertIn("70, 2, 3, 39", prompt)
        self.assertIn("Não substitui avaliação médica", prompt)

    def test_checklist_quality_scores(self):
        text = """
        O recall e a sensibilidade são importantes para reduzir falso negativo.
        Sugere-se triagem e exame complementar.
        Não substitui avaliação médica.
        """
        score = checklist_quality(text)
        self.assertIn("total", score)
        self.assertGreaterEqual(score["foco_em_fn"], 2)
        self.assertGreaterEqual(score["acao_acionavel"], 2)
        self.assertGreaterEqual(score["aviso_segurança"], 2)

    def test_call_gpt_raises_without_api_key(self):
        """Sem OPENAI_API_KEY deve falhar."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                call_gpt("teste")

    @patch("src.llm_interpretation.OpenAI")
    def test_call_gpt_with_mock(self, mock_openai_cls):
        """Mock da OpenAI para não chamar rede."""
        # prepara mock client
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "Resposta mockada do GPT"
        mock_client.responses.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"}):
            out = call_gpt("prompt qualquer", model="gpt-4.1-mini", temperature=0.2)

        self.assertEqual(out, "Resposta mockada do GPT")
        mock_client.responses.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
