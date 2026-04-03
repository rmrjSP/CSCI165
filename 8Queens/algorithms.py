import numpy as np

# ── Problem Definition ────────────────────────────────────────────────────────
# Incremental permutation encoding:
#   chromosome = [r0, r1, r2, r3, r4, r5, r6, r7]
#   ri = row (0-indexed) assigned to queen in column i
#   All rows are distinct → permutation of 0..7
#
# Search space:
#   Unrestricted (rows can repeat): 8^8 = 16,777,216
#   Row-distinct (permutation):     8!  =     40,320  ← we use this
#
# Fitness:
#   Max pairs = C(8,2) = 28
#   Penalty   = number of attacking queen pairs (same row or same diagonal)
#   Fitness   = 28 - penalty    (perfect board = 28)

N_QUEENS   = 8
MAX_PAIRS  = 28   # C(8,2)


# ── Fitness ───────────────────────────────────────────────────────────────────

def fitness(individual):
    """
    Count non-attacking queen pairs.
    Returns 28 for a perfect (no-conflict) board, lower for worse boards.
    """
    penalty = 0
    n = len(individual)
    for c2 in range(1, n):
        for c1 in range(c2):
            row_diff  = abs(individual[c2] - individual[c1])
            col_diff  = c2 - c1  # always positive
            # Same row: row_diff == 0
            # Same diagonal: row_diff == col_diff
            if row_diff == 0 or row_diff == col_diff:
                penalty += 1
    return MAX_PAIRS - penalty


def is_solution(individual):
    """True if no two queens attack each other."""
    return fitness(individual) == MAX_PAIRS


# ── Initialization ────────────────────────────────────────────────────────────

def random_individual(rng):
    """Random permutation of rows 0..7."""
    ind = np.arange(N_QUEENS)
    rng.shuffle(ind)
    return ind


def init_population(pop_size, rng):
    return [random_individual(rng) for _ in range(pop_size)]


# ── Selection ─────────────────────────────────────────────────────────────────

def tournament_select(population, fitnesses, tournament_size, rng):
    """
    Tournament selection: pick tournament_size candidates at random,
    return the one with the highest fitness.
    """
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


# ── Crossover ─────────────────────────────────────────────────────────────────

def cut_and_crossfill(p1, p2, rng):
    """
    Cut-and-crossfill crossover (permutation-safe).
    Matches the class slide recommendation.

    Steps:
      1. Choose a random cut point.
      2. Child starts with p1[0:cut].
      3. Fill the remaining positions with genes from p2
         in order, skipping values already in the child.
    """
    n = len(p1)
    cut = rng.integers(1, n)   # cut point in [1, n-1]

    child = np.empty(n, dtype=int)
    child[:cut] = p1[:cut]

    # Remaining values from p2, in order, skipping duplicates
    used = set(p1[:cut])
    fill_pos = cut
    for gene in p2:
        if gene not in used:
            child[fill_pos] = gene
            used.add(gene)
            fill_pos += 1
            if fill_pos == n:
                break

    return child


# ── Mutation ──────────────────────────────────────────────────────────────────

def swap_mutate(individual, mutation_rate, rng):
    """
    Swap mutation: with probability mutation_rate,
    pick two random positions and swap their row values.
    Preserves the permutation property.
    """
    child = individual.copy()
    if rng.random() < mutation_rate:
        i, j = rng.choice(N_QUEENS, size=2, replace=False)
        child[i], child[j] = child[j], child[i]
    return child


# ── Evolutionary Algorithm ────────────────────────────────────────────────────

def evolutionary_algorithm(pop_size=100, mutation_rate=0.1,
                            max_gens=1000, tournament_size=3,
                            elitism=2, rng=None):
    """
    Full EA loop for the 8-queens problem.

    Args:
        pop_size:        number of individuals in population
        mutation_rate:   probability of swap mutation per child
        max_gens:        maximum generations before stopping
        tournament_size: k for tournament selection
        elitism:         number of best individuals carried to next gen
        rng:             numpy random generator

    Returns:
        best_ind:    best individual found
        best_fit:    fitness of best individual
        history:     best fitness per generation (for convergence plot)
        gen_found:   generation at which perfect solution was found (-1 if not)
    """
    if rng is None:
        rng = np.random.default_rng()

    population = init_population(pop_size, rng)
    fitnesses  = [fitness(ind) for ind in population]

    best_idx  = int(np.argmax(fitnesses))
    best_ind  = population[best_idx].copy()
    best_fit  = fitnesses[best_idx]
    history   = [best_fit]
    gen_found = -1

    for gen in range(max_gens):
        # Early stop if perfect solution found
        if best_fit == MAX_PAIRS:
            gen_found = gen
            break

        # Elitism: carry top-k individuals unchanged
        sorted_idx = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        new_pop = [population[i].copy() for i in sorted_idx[:elitism]]

        # Fill rest of new population with children
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses, tournament_size, rng)
            p2 = tournament_select(population, fitnesses, tournament_size, rng)
            child = cut_and_crossfill(p1, p2, rng)
            child = swap_mutate(child, mutation_rate, rng)
            new_pop.append(child)

        population = new_pop
        fitnesses  = [fitness(ind) for ind in population]

        # Track best
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_ind = population[gen_best_idx].copy()

        history.append(best_fit)

    return best_ind, best_fit, history, gen_found
