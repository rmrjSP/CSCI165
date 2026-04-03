import numpy as np
import csv
import os
import time

from algorithms import TSP, hill_climbing, simulated_annealing, threshold_accepting
from utils import load_or_create_cities, make_rng_list

# ── Config ────────────────────────────────────────────────────────────────────

N_RUNS     = 30
MAX_EVALS  = 100_000     # shared evaluation budget — fair comparison
NEIGHBOR   = "2opt"      # use 2-opt as primary neighbor (swap tested in param sweep)

os.makedirs("../results", exist_ok=True)

# ── Load fixed city dataset ───────────────────────────────────────────────────

coords = load_or_create_cities()
tsp    = TSP(coords)
rngs   = make_rng_list(N_RUNS)

print(f"TSP instance: {tsp.n} cities")
print(f"Running {N_RUNS} trials per algorithm with budget={MAX_EVALS} evals\n")

# ── Define algorithm configurations ──────────────────────────────────────────
# Each entry: (label, callable that takes rng and returns (route, cost, history))

algorithms = {
    "HC_plain": lambda rng: hill_climbing(
        tsp, max_evals=MAX_EVALS, neighbor_fn=NEIGHBOR, restarts=1, rng=rng),

    "HC_restart": lambda rng: hill_climbing(
        tsp, max_evals=MAX_EVALS, neighbor_fn=NEIGHBOR, restarts=5, rng=rng),

    "SA_fast": lambda rng: simulated_annealing(
        tsp, max_evals=MAX_EVALS, T0=1000.0, alpha=0.9990,
        neighbor_fn=NEIGHBOR, rng=rng),

    "SA_slow": lambda rng: simulated_annealing(
        tsp, max_evals=MAX_EVALS, T0=1000.0, alpha=0.9999,
        neighbor_fn=NEIGHBOR, rng=rng),

    "TA": lambda rng: threshold_accepting(
        tsp, max_evals=MAX_EVALS, initial_threshold=50.0,
        n_rounds=200, neighbor_fn=NEIGHBOR, rng=rng),
}

# ── Run all algorithms ────────────────────────────────────────────────────────

raw_results   = []
all_histories = {name: [] for name in algorithms}
best_routes   = {}

for name, algo in algorithms.items():
    run_costs = []
    run_times = []

    for i, rng in enumerate(rngs):
        t0 = time.time()
        best_route, best_cost, history = algo(rng)
        elapsed = time.time() - t0

        run_costs.append(best_cost)
        run_times.append(elapsed)
        all_histories[name].append(history)

        raw_results.append({
            "algorithm":  name,
            "run":        i + 1,
            "best_cost":  round(best_cost, 4),
            "runtime_s":  round(elapsed, 4),
        })

        # Track the absolute best route per algorithm
        if name not in best_routes or best_cost < best_routes[name][1]:
            best_routes[name] = (best_route, best_cost)

    print(f"  {name:<12} | mean={np.mean(run_costs):.2f} | "
          f"best={np.min(run_costs):.2f} | "
          f"std={np.std(run_costs):.2f} | "
          f"avg time={np.mean(run_times)*1000:.0f}ms")

# ── Save raw results ──────────────────────────────────────────────────────────

with open("../results/raw_runs.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=raw_results[0].keys())
    writer.writeheader()
    writer.writerows(raw_results)

# ── Save summary ──────────────────────────────────────────────────────────────

summary = []
for name in algorithms:
    runs  = [r for r in raw_results if r["algorithm"] == name]
    costs = [r["best_cost"] for r in runs]
    times = [r["runtime_s"] for r in runs]
    summary.append({
        "algorithm":    name,
        "mean_cost":    round(float(np.mean(costs)), 2),
        "std_cost":     round(float(np.std(costs)), 2),
        "best_cost":    round(float(np.min(costs)), 2),
        "worst_cost":   round(float(np.max(costs)), 2),
        "mean_time_ms": round(float(np.mean(times)) * 1000, 1),
    })

with open("../results/summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

# ── Save best routes and histories ────────────────────────────────────────────

with open("../results/best_solutions.txt", "w") as f:
    f.write("Best routes found per algorithm\n")
    f.write("=" * 40 + "\n\n")
    for name, (route, cost) in best_routes.items():
        f.write(f"{name}: cost = {cost:.4f}\n")
        f.write(f"  route = {list(route)}\n\n")

np.save("../results/histories.npy",    all_histories, allow_pickle=True)
np.save("../results/best_routes.npy",  best_routes,   allow_pickle=True)

print("\nResults saved to ../results/")
