import numpy as np
import csv
import os
import time
from algorithms import (
    rastrigin,
    gradient_descent_fixed,
    gradient_descent_decaying,
    gradient_descent_momentum,
    simulated_annealing,
    DOMAIN_MIN, DOMAIN_MAX
)

# ── Config ────────────────────────────────────────────────────────────────────

N_RUNS       = 30       # number of random starting points
MAX_ITER     = 5000     # shared iteration budget for all algorithms
SUCCESS_TOL  = 1e-3     # f(x) < this counts as "found global minimum"
RANDOM_SEED  = 42

# Algorithm parameters (tuned from guide recommendations)
GD_FIXED_ALPHA    = 0.01
GD_DECAY_ALPHA0   = 0.1
GD_DECAY_RATE     = 0.001
GD_MOMENTUM_ALPHA = 0.01
GD_MOMENTUM_BETA  = 0.9
SA_T0             = 10.0
SA_ALPHA          = 0.995
SA_T_MIN          = 1e-4
SA_STEP_RADIUS    = 0.5

# ── Generate shared starting points ──────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)
starting_points = rng.uniform(DOMAIN_MIN, DOMAIN_MAX, size=(N_RUNS, 2))

# ── Run experiments ───────────────────────────────────────────────────────────

algorithms = {
    "GD_Fixed":    lambda x0: gradient_descent_fixed(x0, alpha=GD_FIXED_ALPHA, max_iter=MAX_ITER),
    "GD_Decaying": lambda x0: gradient_descent_decaying(x0, alpha0=GD_DECAY_ALPHA0, decay=GD_DECAY_RATE, max_iter=MAX_ITER),
    "GD_Momentum": lambda x0: gradient_descent_momentum(x0, alpha=GD_MOMENTUM_ALPHA, beta=GD_MOMENTUM_BETA, max_iter=MAX_ITER),
    "SA":          lambda x0: simulated_annealing(x0, T0=SA_T0, alpha=SA_ALPHA, T_min=SA_T_MIN, max_iter=MAX_ITER, step_radius=SA_STEP_RADIUS),
}

# Store all raw results and all convergence histories
raw_results  = []    # one row per (algorithm, run)
all_histories = {name: [] for name in algorithms}

print(f"Running {N_RUNS} trials per algorithm...\n")

for name, algo in algorithms.items():
    run_times = []
    for i, x0 in enumerate(starting_points):
        t_start = time.time()
        best_x, best_f, history = algo(x0.copy())
        elapsed = time.time() - t_start

        success = int(best_f < SUCCESS_TOL)
        dist_to_opt = np.linalg.norm(best_x)  # distance from (0,0)

        raw_results.append({
            "algorithm":   name,
            "run":         i + 1,
            "best_f":      round(best_f, 6),
            "x1":          round(best_x[0], 6),
            "x2":          round(best_x[1], 6),
            "dist_to_opt": round(dist_to_opt, 6),
            "success":     success,
            "runtime_s":   round(elapsed, 4),
            "iters":       len(history) - 1,
        })
        all_histories[name].append(history)
        run_times.append(elapsed)

    # Print a quick summary per algorithm
    finals = [r["best_f"] for r in raw_results if r["algorithm"] == name]
    successes = [r["success"] for r in raw_results if r["algorithm"] == name]
    print(f"  {name:15s} | mean f={np.mean(finals):.4f} | "
          f"best f={np.min(finals):.6f} | "
          f"success={sum(successes)}/{N_RUNS} | "
          f"avg time={np.mean(run_times)*1000:.1f}ms")

# ── Save raw results ──────────────────────────────────────────────────────────

os.makedirs("../results", exist_ok=True)

with open("../results/raw_runs.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=raw_results[0].keys())
    writer.writeheader()
    writer.writerows(raw_results)

# ── Save summary stats ────────────────────────────────────────────────────────

summary = []
for name in algorithms:
    runs = [r for r in raw_results if r["algorithm"] == name]
    finals = [r["best_f"] for r in runs]
    times  = [r["runtime_s"] for r in runs]
    succs  = [r["success"] for r in runs]
    summary.append({
        "algorithm":      name,
        "mean_f":         round(float(np.mean(finals)), 6),
        "std_f":          round(float(np.std(finals)), 6),
        "best_f":         round(float(np.min(finals)), 6),
        "worst_f":        round(float(np.max(finals)), 6),
        "success_rate":   f"{sum(succs)}/{N_RUNS}",
        "mean_time_ms":   round(float(np.mean(times)) * 1000, 2),
    })

with open("../results/summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

# ── Save best solutions ───────────────────────────────────────────────────────

with open("../results/best_solutions.txt", "w") as f:
    f.write("Best solutions found per algorithm\n")
    f.write("=" * 45 + "\n\n")
    for name in algorithms:
        runs = [r for r in raw_results if r["algorithm"] == name]
        best = min(runs, key=lambda r: r["best_f"])
        f.write(f"{name}\n")
        f.write(f"  x1 = {best['x1']},  x2 = {best['x2']}\n")
        f.write(f"  f(x) = {best['best_f']}\n")
        f.write(f"  dist to (0,0) = {best['dist_to_opt']}\n\n")

# Also save histories for use by plots.py
np.save("../results/histories.npy", all_histories, allow_pickle=True)
np.save("../results/starting_points.npy", starting_points)

print("\nResults saved to ../results/")
