from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


@dataclass
class GAConfigTree:
    # GA
    population_size: int = 50
    n_generations: int = 20
    mut_prob: float = 0.20
    cx_prob: float = 0.70
    tournament_k: int = 3
    cv_folds: int = 5     
    random_state: int = 42

    # Pesos do fitness (contexto clínico: recall é prioridade)
    w_recall: float = 0.60
    w_f1: float = 0.30
    w_acc: float = 0.10

    # Penalização de complexidade (reduz overfitting)
    complexity_penalty: float = 0.03


# ----------------------------------------------------------------------
# ESPAÇO DE BUSCA (genes)
# ----------------------------------------------------------------------

TREE_SPACE = {
    # antes: [None, 2,3,4,5,6,8,10]
    # agora: limite superior menor para reduzir árvores profundas (overfitting)
    "max_depth": [2, 3, 4, 5, 6, 7, 8, None],

    # antes: [2,3,4,5,6,8,10]
    # agora: começa em 5 para evitar splits com poucos exemplos (menos overfitting)
    "min_samples_split": [2, 3, 5, 8, 10, 12],

    # antes: [1,2,3,4,5]
    # agora: começa em 2 para evitar folhas “isoladas”
    "min_samples_leaf": [1, 2, 3, 4, 5],

    # antes: ["gini", "entropy", "log_loss"]
    "criterion": ["gini", "entropy"],

    # antes: ["best", "random"]
    "splitter": ["best"],

    "max_features": [None, "sqrt", "log2"],
    "class_weight": [None, "balanced"],
}


def _random_individual(rng: random.Random) -> Dict[str, Any]:
    """
    Cria um indivíduo (um conjunto de hiperparâmetros da árvore).
    """
    return {
        "max_depth": rng.choice(TREE_SPACE["max_depth"]),
        "min_samples_split": rng.choice(TREE_SPACE["min_samples_split"]),
        "min_samples_leaf": rng.choice(TREE_SPACE["min_samples_leaf"]),
        "criterion": rng.choice(TREE_SPACE["criterion"]),
        "splitter": rng.choice(TREE_SPACE["splitter"]),
        "max_features": rng.choice(TREE_SPACE["max_features"]),
        "class_weight": rng.choice(TREE_SPACE["class_weight"]),
    }


def _build_model(params: Dict[str, Any], random_state: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        random_state=random_state,
        **params
    )


def _complexity_cost(params: Dict[str, Any]) -> float:
    """
    Penalização simples de complexidade para reduzir overfitting.

    Intuição:
    max_depth maior => árvore mais complexa
    min_samples_leaf pequeno => mais chance de memorizar dados
    """
    depth = params.get("max_depth", None)
    depth_cost = float(depth) if depth is not None else 9.0  

    leaf = int(params["min_samples_leaf"])
    leaf_cost = 1.0 / leaf  
    return (0.02 * depth_cost) + (0.03 * leaf_cost)


def _fitness(
    X_train,
    y_train,
    params: Dict[str, Any],
    config: GAConfigTree
) -> float:
    """
    Fitness baseado em validação cruzada estratificada.
    """
    model = _build_model(params, random_state=config.random_state)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    scoring = {
        "acc": "accuracy",
        "recall": "recall",
        "f1": "f1",
    }

    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    acc = float(np.mean(scores["test_acc"]))
    rec = float(np.mean(scores["test_recall"]))
    f1 = float(np.mean(scores["test_f1"]))

    base_fitness = (config.w_recall * rec) + (config.w_f1 * f1) + (config.w_acc * acc)

    # Penalização por complexidade 
    penalty = config.complexity_penalty * _complexity_cost(params)

    return base_fitness - penalty


def _tournament_selection(
    population: List[Dict[str, Any]],
    fitnesses: List[float],
    rng: random.Random,
    k: int
) -> Dict[str, Any]:
    idxs = rng.sample(range(len(population)), k)
    best_idx = max(idxs, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def _crossover(p1: Dict[str, Any], p2: Dict[str, Any], rng: random.Random) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    c1, c2 = p1.copy(), p2.copy()
    for key in c1.keys():
        if rng.random() < 0.5:
            c1[key], c2[key] = c2[key], c1[key]
    return c1, c2


def _mutate(ind: Dict[str, Any], rng: random.Random, mut_prob: float) -> Dict[str, Any]:
   
    new_ind = ind.copy()

    def maybe_mutate_gene(key: str, options: List[Any]):
        if rng.random() < mut_prob:
            new_ind[key] = rng.choice(options)

    maybe_mutate_gene("max_depth", TREE_SPACE["max_depth"])
    maybe_mutate_gene("min_samples_split", TREE_SPACE["min_samples_split"])
    maybe_mutate_gene("min_samples_leaf", TREE_SPACE["min_samples_leaf"])
    maybe_mutate_gene("criterion", TREE_SPACE["criterion"])
    maybe_mutate_gene("splitter", TREE_SPACE["splitter"])
    maybe_mutate_gene("max_features", TREE_SPACE["max_features"])
    maybe_mutate_gene("class_weight", TREE_SPACE["class_weight"])

    return new_ind


def optimize_tree_with_ga(
    X_train,
    y_train,
    config: Optional[GAConfigTree] = None
) -> Tuple[DecisionTreeClassifier, Dict[str, Any], pd.DataFrame]:

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
        fitnesses = [_fitness(X_train, y_train, ind, config) for ind in population]

        gen_min = float(np.min(fitnesses))
        gen_avg = float(np.mean(fitnesses))
        gen_max = float(np.max(fitnesses))

        best_idx = int(np.argmax(fitnesses))
        gen_best_ind = population[best_idx].copy()
        gen_best_fit = float(fitnesses[best_idx])

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_params = gen_best_ind.copy()

        logger.info(
            "Gen %d | min=%.4f avg=%.4f max=%.4f | best=%s",
            gen, gen_min, gen_avg, gen_max, gen_best_ind
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
            "best_class_weight": gen_best_ind["class_weight"],
        })

        # 3) Nova população com elitismo simples (mantém o melhor)
        new_population = [gen_best_ind.copy()]

        while len(new_population) < config.population_size:
            p1 = _tournament_selection(population, fitnesses, rng, config.tournament_k)
            p2 = _tournament_selection(population, fitnesses, rng, config.tournament_k)

            if rng.random() < config.cx_prob:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

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
