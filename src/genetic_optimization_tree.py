from sklearn.tree import DecisionTreeClassifier

TREE_SPACE = {
    "max_depth": (2, 20),            
    "min_samples_split": (2, 20),     
    "min_samples_leaf": (1, 10),      
    "criterion": ["gini", "entropy"], 
}


def _decode_tree_individual(individual: List[int]) -> Dict[str, Any]:
   
    max_depth_gene = int(individual[0])
    max_depth = None if max_depth_gene == 0 else max_depth_gene

    min_samples_split = int(individual[1])
    min_samples_leaf = int(individual[2])
    criterion = TREE_SPACE["criterion"][int(individual[3])]

    return {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
        "random_state": 42
    }


def _evaluate_tree_fitness(
    individual: List[int],
    X: np.ndarray,
    y: np.ndarray,
    config: GAConfig,
) -> Tuple[float]:
    params = _decode_tree_individual(individual)
    model = DecisionTreeClassifier(**params)

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    scoring = {"acc": "accuracy", "recall": "recall", "f1": "f1"}
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    acc = float(np.mean(scores["test_acc"]))
    rec = float(np.mean(scores["test_recall"]))
    f1 = float(np.mean(scores["test_f1"]))

    fitness = (config.w_recall * rec) + (config.w_f1 * f1) + (config.w_acc * acc)
    return (fitness,)


def _mutate_tree_individual(individual: List[int], indpb: float = 0.2) -> Tuple[List[int]]:
    
    # gene 0: max_depth_gene (0 ou 2..20)
    if random.random() < indpb:
        # 20% chance de virar None (0), senão vira um depth válido
        if random.random() < 0.2:
            individual[0] = 0
        else:
            low, high = TREE_SPACE["max_depth"]
            individual[0] = random.randint(low, high)

    # gene 1: min_samples_split
    if random.random() < indpb:
        low, high = TREE_SPACE["min_samples_split"]
        individual[1] = random.randint(low, high)

    # gene 2: min_samples_leaf
    if random.random() < indpb:
        low, high = TREE_SPACE["min_samples_leaf"]
        individual[2] = random.randint(low, high)

    # gene 3: criterion_idx
    if random.random() < indpb:
        individual[3] = random.randrange(len(TREE_SPACE["criterion"]))

    return (individual,)


def optimize_decision_tree_with_ga(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[GAConfig] = None,
) -> Tuple[DecisionTreeClassifier, Dict[str, Any], pd.DataFrame]:
    
    if config is None:
        config = GAConfig()

    random.seed(config.random_state)
    np.random.seed(config.random_state)

    # Criadores DEAP (evita erro se reexecutar no notebook)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # geradores de genes
    def gene_max_depth():
        # 20% chance de None (0)
        if random.random() < 0.2:
            return 0
        low, high = TREE_SPACE["max_depth"]
        return random.randint(low, high)

    def gene_min_samples_split():
        low, high = TREE_SPACE["min_samples_split"]
        return random.randint(low, high)

    def gene_min_samples_leaf():
        low, high = TREE_SPACE["min_samples_leaf"]
        return random.randint(low, high)

    def gene_criterion_idx():
        return random.randrange(len(TREE_SPACE["criterion"]))

    # indivíduo: [max_depth_gene, min_samples_split, min_samples_leaf, criterion_idx]
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (gene_max_depth, gene_min_samples_split, gene_min_samples_leaf, gene_criterion_idx),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", _evaluate_tree_fitness, X=X, y=y, config=config)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", _mutate_tree_individual, indpb=0.2)

    pop = toolbox.population(n=config.population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    history_rows = []

    # avalia população inicial
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    for gen in range(1, config.n_generations + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # crossover
        for c1, c2 in zip(offspring[0::2], offspring[1::2]):
            if random.random() < config.cx_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # mutation
        for mutant in offspring:
            if random.random() < config.mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # reavalia
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        record = stats.compile(pop)
        best_ind = tools.selBest(pop, 1)[0]
        best_params = _decode_tree_individual(best_ind)

        history_rows.append({
            "generation": gen,
            "fitness_min": record["min"],
            "fitness_avg": record["avg"],
            "fitness_max": record["max"],
            "best_max_depth": best_params["max_depth"],
            "best_min_samples_split": best_params["min_samples_split"],
            "best_min_samples_leaf": best_params["min_samples_leaf"],
            "best_criterion": best_params["criterion"],
        })

        logger.info(
            "Tree Gen %d | min=%.4f avg=%.4f max=%.4f | best=%s",
            gen, record["min"], record["avg"], record["max"], best_params
        )

    best_ind = tools.selBest(pop, 1)[0]
    best_params = _decode_tree_individual(best_ind)

    best_model = DecisionTreeClassifier(**best_params)
    history_df = pd.DataFrame(history_rows)

    return best_model, best_params, history_df
