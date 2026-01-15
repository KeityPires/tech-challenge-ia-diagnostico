
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from deap import base, creator, tools
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier


logger = logging.getLogger(__name__)



# Configuração do GA
@dataclass
class GAConfig:
    # GA
    population_size: int = 50
    n_generations: int = 20
    cx_prob: float = 0.7
    mut_prob: float = 0.2
    tournament_size: int = 3
    random_state: int = 42

    # Avaliação (fitness)
    cv_folds: int = 5
   
    # Peso das métricas (prioriza recall por contexto médico)
    w_recall: float = 0.5
    w_f1: float = 0.3
    w_acc: float = 0.2


# Espaço de busca do KNN (genes)
KNN_SPACE = {
    "n_neighbors": (3, 25),  
    "weights": ["uniform", "distance"],  
    "metric": ["euclidean", "manhattan"],  
}


def _decode_individual(individual: List[int]) -> Dict[str, Any]:
    
    n_neighbors = int(individual[0])
    weights = KNN_SPACE["weights"][int(individual[1])]
    metric = KNN_SPACE["metric"][int(individual[2])]

    params = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
    }
    return params


def _evaluate_knn_fitness(
    individual: List[int],
    X: np.ndarray,
    y: np.ndarray,
    config: GAConfig,
) -> Tuple[float]:
   
    params = _decode_individual(individual)
    model = KNeighborsClassifier(**params)

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    scoring = {
        "acc": "accuracy",
        "recall": "recall",
        "f1": "f1",
    }

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    acc = float(np.mean(scores["test_acc"]))
    rec = float(np.mean(scores["test_recall"]))
    f1 = float(np.mean(scores["test_f1"]))

    fitness = (config.w_recall * rec) + (config.w_f1 * f1) + (config.w_acc * acc)

    return (fitness,)


def _mutate_individual(individual: List[int], indpb: float = 0.2) -> Tuple[List[int]]:
    
    # gene 0: n_neighbors
    if random.random() < indpb:
        low, high = KNN_SPACE["n_neighbors"]
        individual[0] = random.randint(low, high)

    # gene 1: weights_idx
    if random.random() < indpb:
        individual[1] = random.randrange(len(KNN_SPACE["weights"]))

    # gene 2: metric_idx
    if random.random() < indpb:
        individual[2] = random.randrange(len(KNN_SPACE["metric"]))

    return (individual,)


def optimize_knn_with_ga(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[GAConfig] = None,
) -> Tuple[KNeighborsClassifier, Dict[str, Any], pd.DataFrame]:
    
    if config is None:
        config = GAConfig()

    # Seed para reprodutibilidade
    random.seed(config.random_state)
    np.random.seed(config.random_state)

    # Criadores DEAP (evita erro ao reexecutar no notebook)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Gene generators
    def gene_n_neighbors():
        low, high = KNN_SPACE["n_neighbors"]
        return random.randint(low, high)

    def gene_weights_idx():
        return random.randrange(len(KNN_SPACE["weights"]))

    def gene_metric_idx():
        return random.randrange(len(KNN_SPACE["metric"]))

    # Individual: [n_neighbors, weights_idx, metric_idx]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (gene_n_neighbors, gene_weights_idx, gene_metric_idx), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operators
    toolbox.register("evaluate", _evaluate_knn_fitness, X=X, y=y, config=config)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)  # cruzamento uniforme
    toolbox.register("mutate", _mutate_individual, indpb=0.2)

    # Inicializa população
    pop = toolbox.population(n=config.population_size)

    # Estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    history_rows = []

    # Avalia população inicial
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    for gen in range(1, config.n_generations + 1):
        # Seleção
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Cruzamento
        for c1, c2 in zip(offspring[0::2], offspring[1::2]):
            if random.random() < config.cx_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutação
        for mutant in offspring:
            if random.random() < config.mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Reavalia indivíduos alterados
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # Substitui população
        pop[:] = offspring

        # Logging de estatísticas
        record = stats.compile(pop)
        best_ind = tools.selBest(pop, 1)[0]
        best_params = _decode_individual(best_ind)

        history_rows.append({
            "generation": gen,
            "fitness_min": record["min"],
            "fitness_avg": record["avg"],
            "fitness_max": record["max"],
            "best_n_neighbors": best_params["n_neighbors"],
            "best_weights": best_params["weights"],
            "best_metric": best_params["metric"],
        })

        logger.info(
            "Gen %d | min=%.4f avg=%.4f max=%.4f | best=%s",
            gen, record["min"], record["avg"], record["max"], best_params
        )

    # Melhor indivíduo final
    best_ind = tools.selBest(pop, 1)[0]
    best_params = _decode_individual(best_ind)

    best_model = KNeighborsClassifier(**best_params)
    history_df = pd.DataFrame(history_rows)

    return best_model, best_params, history_df
