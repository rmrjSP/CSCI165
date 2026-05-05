import numpy as np

# ── TSP Problem ───────────────────────────────────────────────────────────────
# 100 cities with random 2D coordinates in [0, 100] x [0, 100]
# Solution: permutation of city indices [0 .. n-1]
# Objective: total closed-tour distance (minimize)

class TSP:
    def __init__(self, coords):
        """
        Args:
            coords: np.array of shape (n, 2) — x,y coordinates for each city
        """
        self.coords = np.array(coords)
        self.n = len(coords)
        # Precompute full distance matrix for speed
        self._dist_matrix = self._build_dist_matrix()

    def _build_dist_matrix(self):
        """Euclidean distance between every pair of cities."""
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def route_cost(self, route):
        """Total length of a closed tour (last city connects back to first)."""
        r = np.array(route)
        # Sum distance for each consecutive pair, including wrap-around
        return self._dist_matrix[r[:-1], r[1:]].sum() + self._dist_matrix[r[-1], r[0]]

    def random_route(self, rng=None):
        """Return a random permutation of all city indices."""
        if rng is None:
            rng = np.random.default_rng()
        route = np.arange(self.n)
        rng.shuffle(route)
        return route

    # ── Neighborhood operators ────────────────────────────────────────────────

    def neighbor_swap(self, route, rng=None):
        """Swap two randomly chosen cities in the route."""
        if rng is None:
            rng = np.random.default_rng()
        new_route = route.copy()
        i, j = rng.choice(self.n, size=2, replace=False)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def neighbor_two_opt(self, route, rng=None):
        """
        2-opt move: reverse a segment of the route between positions i and j.
        Stronger than swap — standard for TSP local search.
        """
        if rng is None:
            rng = np.random.default_rng()
        new_route = route.copy()
        i, j = sorted(rng.choice(self.n, size=2, replace=False))
        new_route[i:j+1] = new_route[i:j+1][::-1]
        return new_route


# ── Algorithm 1: Hill Climbing ────────────────────────────────────────────────

def hill_climbing(tsp, max_evals=100_000, neighbor_fn="2opt",
                  restarts=1, rng=None):
    """
    Hill climbing: accept only improving moves.

    Args:
        tsp:          TSP instance
        max_evals:    total neighbor evaluations budget (shared with SA/TA)
        neighbor_fn:  "swap" or "2opt"
        restarts:     number of random restarts (1 = plain hill climbing)
        rng:          numpy random generator for reproducibility

    Returns:
        best_route:  best tour found
        best_cost:   length of best tour
        history:     best cost after each evaluation (for convergence plot)
    """
    if rng is None:
        rng = np.random.default_rng()

    neighbor = tsp.neighbor_two_opt if neighbor_fn == "2opt" else tsp.neighbor_swap

    best_route = None
    best_cost  = float("inf")
    history    = []

    evals_per_restart = max_evals // restarts

    for _ in range(restarts):
        route = tsp.random_route(rng)
        cost  = tsp.route_cost(route)

        for _ in range(evals_per_restart):
            new_route = neighbor(route, rng)
            new_cost  = tsp.route_cost(new_route)

            # Only accept strictly improving moves
            if new_cost < cost:
                route = new_route
                cost  = new_cost

            history.append(best_cost if cost >= best_cost else cost)

            if cost < best_cost:
                best_cost  = cost
                best_route = route.copy()

    return best_route, best_cost, history


# ── Algorithm 2: Simulated Annealing ─────────────────────────────────────────

def simulated_annealing(tsp, max_evals=100_000, T0=1000.0, alpha=0.9995,
                        T_min=1e-3, neighbor_fn="2opt", rng=None):
    """
    Simulated annealing for TSP.

    Accepts worse moves with probability exp(-delta / T).
    Temperature decays exponentially: T = T * alpha each step.

    Args:
        tsp:          TSP instance
        max_evals:    neighbor evaluation budget
        T0:           initial temperature
        alpha:        cooling rate (closer to 1 = slower cooling)
        T_min:        stop early if T drops below this
        neighbor_fn:  "swap" or "2opt"
        rng:          numpy random generator

    Returns: same as hill_climbing
    """
    if rng is None:
        rng = np.random.default_rng()

    neighbor = tsp.neighbor_two_opt if neighbor_fn == "2opt" else tsp.neighbor_swap

    route      = tsp.random_route(rng)
    cost       = tsp.route_cost(route)
    best_route = route.copy()
    best_cost  = cost
    history    = []
    T          = T0

    for _ in range(max_evals):
        if T < T_min:
            history.append(best_cost)
            continue

        new_route = neighbor(route, rng)
        new_cost  = tsp.route_cost(new_route)
        delta     = new_cost - cost

        # Accept if better, or probabilistically if worse
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            route = new_route
            cost  = new_cost

        if cost < best_cost:
            best_cost  = cost
            best_route = route.copy()

        history.append(best_cost)
        T *= alpha

    return best_route, best_cost, history


# ── Algorithm 3: Threshold Accepting ─────────────────────────────────────────

def threshold_accepting(tsp, max_evals=100_000, initial_threshold=50.0,
                        n_rounds=200, neighbor_fn="2opt", rng=None):
    """
    Threshold accepting for TSP.

    Like SA but acceptance is deterministic:
    accept if new_cost - current_cost <= threshold  (no randomness).
    Threshold decreases linearly from initial_threshold to 0 over n_rounds,
    acting as full exploration early and pure hill climbing at the end.

    Args:
        tsp:               TSP instance
        max_evals:         neighbor evaluation budget
        initial_threshold: starting acceptance threshold
        n_rounds:          number of rounds over which threshold decays to 0
        neighbor_fn:       "swap" or "2opt"
        rng:               numpy random generator

    Returns: same as hill_climbing
    """
    if rng is None:
        rng = np.random.default_rng()

    neighbor = tsp.neighbor_two_opt if neighbor_fn == "2opt" else tsp.neighbor_swap

    route      = tsp.random_route(rng)
    cost       = tsp.route_cost(route)
    best_route = route.copy()
    best_cost  = cost
    history    = []

    steps_per_round = max_evals // n_rounds

    for r in range(n_rounds):
        # Linearly decay: large threshold early → 0 at end (pure hill climbing)
        threshold = initial_threshold * (1.0 - r / n_rounds)

        for _ in range(steps_per_round):
            new_route = neighbor(route, rng)
            new_cost  = tsp.route_cost(new_route)
            delta     = new_cost - cost

            # Deterministic: accept if better OR within threshold
            if delta <= threshold:
                route = new_route
                cost  = new_cost

            if cost < best_cost:
                best_cost  = cost
                best_route = route.copy()

            history.append(best_cost)

    return best_route, best_cost, history
