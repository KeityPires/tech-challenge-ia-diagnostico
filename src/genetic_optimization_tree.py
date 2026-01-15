from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


@dataclass
class GAConfigTree:
    population_size: int = 20
    n_generations: int = 10
    mut_prob: float = 0.10
    cx_prob: float = 0.70
    tournament_k: int = 3
    cv_folds: int = 5
    random_state: int = 42

    # Pesos para a função fitness (ajuste conforme seu foco clínico)
    # Em saúde, muitas vezes o recall (sensibilidade) é mais importante.
    w_recall: float = 0.50
    w_f1: float = 0.40
    w_acc: float = 0.10


def _random_individual(rng: random.Random) -> Dict[str, Any]:
    """
    Cria um indivíduo (um conjunto de hiperparâmetros da árvore).
    Genes escolhidos para serem "otimizáveis" e comuns em tuning.
    """
    max_depth = rng.choice([None, 2, 3, 4, 5, 6, 8, 10])
    min_samples_split = rng.choice([2, 3, 4, 5, 6, 8, 10])
    min_samples_leaf = rng.choice([1, 2, 3, 4, 5])
    criterion = rng.choice(["gini", "entropy", "log_loss"])
    splitter = rng.choice(["best", "random"])

    # max_features pode melhorar generalização (reduzir overfitting).
    max_features = rng.choice([None, "sqrt", "log2"])

    return {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
        "splitter": splitter,
        "max_features": max_features,
    }


def _build_model(params: Dict[str, Any], random_state: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        random_state=random_state,
        **params
    )


def _fitness(
    X_train,
    y_train,
    params: Dict[str, Any],
    config: GAConfigTree
) -> float:
    """
    Calcula fitness baseado em validação cruzada estratificada no TREINO.
    Fitness = combinação ponderada de recall + f1 + accuracy.
    """
    model = _build_model(params, random_state=config.random_state)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    # cross_val_score retorna média do score por fold
    acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    rec = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall").mean()
    f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()

    return (config.w_recall * rec) + (config.w_f1 * f1) + (config.w_acc * acc)


def _tournament_selection(population: List[Dict[str, Any]], fitnesses: List[float], rng: random.Random, k: int) -> Dict[str, Any]:
    """
    Seleciona 1 indivíduo por torneio:
    escolhe k aleatórios e retorna o com maior fitness.
    """
    idxs = rng.sample(range(len(population)), k)
    best_idx = max(idxs, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def _crossover(p1: Dict[str, Any], p2: Dict[str, Any], rng: random.Random) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Crossover simples: para cada gene, troca com 50% de chance.
    """
    c1, c2 = p1.copy(), p2.copy()
    for key in c1.keys():
        if rng.random() < 0.5:
            c1[key], c2[key] = c2[key], c1[key]
    return c1, c2


def _mutate(ind: Dict[str, Any], rng: random.Random, mut_prob: float) -> Dict[str, Any]:
    """
    Mutação: com probabilidade mut_prob, altera um ou mais genes.
    """
    new_ind = ind.copy()

    def maybe_mutate_gene(key: str, options: List[Any]):
        if rng.random() < mut_prob:
            new_ind[key] = rng.choice(options)

    maybe_mutate_gene("max_depth", [None, 2, 3, 4, 5, 6, 8, 10])
    maybe_mutate_gene("min_samples_split", [2, 3, 4, 5, 6, 8, 10])
    maybe_mutate_gene("min_samples_leaf", [1, 2, 3, 4, 5])
    maybe_mutate_gene("criterion", ["gini", "entropy", "log_loss"])
    maybe_mutate_gene("splitter", ["best", "random"])
    maybe_mutate_gene("max_features", [None, "sqrt", "log2"])

    return new_ind


def optimize_tree_with_ga(
    X_train,
    y_train,
    config: Optional[GAConfigTree] = None
) -> Tuple[DecisionTreeClassifier, Dict[str, Any], pd.DataFrame]:
    """
    Executa o AG para encontrar os melhores hiperparâmetros da Decision Tree.

    Retorna:
    - best_model: modelo treinado com melhores hiperparâmetros no treino completo
    - best_params: dicionário com hiperparâmetros vencedores
    - history: DataFrame com estatísticas por geração (min/avg/max + best_*)
    """
    if config is None:
        config = GAConfigTree()

    rng = random.Random(config.random_state)

    # 1) Inicializa população
    population = [_random_individual(rng) for _ in range(config.population_size)]

    history_rows = []
    best_params: Dict[str, Any] = {}
    best_fit = -np.inf

    # 2) Evolui por gerações
    for gen in range(1, config.n_generations + 1):
        fitnesses = [
            _fitness(X_train, y_train, ind, config)
            for ind in population
        ]

        gen_min = float(np.min(fitnesses))
        gen_avg = float(np.mean(fitnesses))
        gen_max = float(np.max(fitnesses))

        # Melhor da geração
        best_idx = int(np.argmax(fitnesses))
        gen_best_ind = population[best_idx].copy()
        gen_best_fit = fitnesses[best_idx]

        # Melhor global
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_params = gen_best_ind.copy()

        logger.info(
            f"Gen {gen} | min={gen_min:.4f} avg={gen_avg:.4f} max={gen_max:.4f} | best={gen_best_ind}"
        )

        history_rows.append({
            "generation": gen,
            "fitness_min": gen_min,
            "fitness_avg": gen_avg,
            "fitness_max": gen_max,
            "best_max_depth": gen_best_ind["max_depth"],
            "best_min_samples_split": gen_best_ind["min_samples_split"],
            "best_min_samples_leaf": gen_best_ind["min_samples_leaf"],
            "best_criterion": gen_best_ind["criterion"],
            "best_splitter": gen_best_ind["splitter"],
            "best_max_features": gen_best_ind["max_features"],
        })

        # 3) Nova população (elitismo simples: mantém o melhor)
        new_population = [gen_best_ind.copy()]

        while len(new_population) < config.population_size:
            # seleção
            p1 = _tournament_selection(population, fitnesses, rng, config.tournament_k)
            p2 = _tournament_selection(population, fitnesses, rng, config.tournament_k)

            # crossover
            if rng.random() < config.cx_prob:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # mutação
            c1 = _mutate(c1, rng, config.mut_prob)
            c2 = _mutate(c2, rng, config.mut_prob)

            new_population.append(c1)
            if len(new_population) < config.population_size:
                new_population.append(c2)

        population = new_population

    # 4) Treina modelo final com melhores hiperparâmetros no treino completo
    best_model = _build_model(best_params, random_state=config.random_state)
    best_model.fit(X_train, y_train)

    history = pd.DataFrame(history_rows)
    return best_model, best_params, history
