import numpy as np
import csv
import os
import time
from algorithms import evolutionary_algorithm, MAX_PAIRS

# ── Config ────────────────────────────────────────────────────────────────────

N_RUNS      = 30
BASE_SEED   = 42

os.makedirs("../results", exist_ok=True)

# ── Experiment definitions ────────────────────────────────────────────────────
# Three sweeps from the guide:
#   1. Population size  (fixing mutation=0.1, tournament=3)
#   2. Mutation rate    (fixing pop=100,     tournament=3)
#   3. Tournament size  (fixing pop=100,     mutation=0.1)
#
# Plus one "best config" run for the final comparison.

experiments = [
    # --- Population size sweep ---
    {"label": "pop=20",   "pop_size": 20,  "mutation_rate": 0.10, "tournament_size": 3},
    {"label": "pop=50",   "pop_size": 50,  "mutation_rate": 0.10, "tournament_size": 3},
    {"label": "pop=100",  "pop_size": 100, "mutation_rate": 0.10, "tournament_size": 3},

    # --- Mutation rate sweep ---
    {"label": "mut=0.05", "pop_size": 100, "mutation_rate": 0.05, "tournament_size": 3},
    {"label": "mut=0.10", "pop_size": 100, "mutation_rate": 0.10, "tournament_size": 3},
    {"label": "mut=0.20", "pop_size": 100, "mutation_rate": 0.20, "tournament_size": 3},

    # --- Tournament size sweep ---
    {"label": "tour=2",   "pop_size": 100, "mutation_rate": 0.10, "tournament_size": 2},
    {"label": "tour=3",   "pop_size": 100, "mutation_rate": 0.10, "tournament_size": 3},
    {"label": "tour=5",   "pop_size": 100, "mutation_rate": 0.10, "tournament_size": 5},
]

# ── Run all experiments ───────────────────────────────────────────────────────

raw_results   = []
all_histories = {}

print(f"Running {N_RUNS} trials per configuration...\n")

for exp in experiments:
    label         = exp["label"]
    successes     = 0
    gens_to_solve = []
    final_fits    = []
    histories     = []

    for run in range(N_RUNS):
        rng = np.random.default_rng(BASE_SEED + run)

        t0 = time.time()
        best_ind, best_fit, history, gen_found = evolutionary_algorithm(
            pop_size       = exp["pop_size"],
            mutation_rate  = exp["mutation_rate"],
            max_gens       = 1000,
            tournament_size= exp["tournament_size"],
            elitism        = 2,
            rng            = rng,
        )
        elapsed = time.time() - t0

        solved = int(best_fit == MAX_PAIRS)
        successes += solved
        final_fits.append(best_fit)
        histories.append(history)

        if solved:
            gens_to_solve.append(gen_found)

        raw_results.append({
            "config":     label,
            "run":        run + 1,
            "best_fit":   best_fit,
            "solved":     solved,
            "gen_found":  gen_found,
            "runtime_s":  round(elapsed, 4),
            "pop_size":   exp["pop_size"],
            "mutation":   exp["mutation_rate"],
            "tournament": exp["tournament_size"],
        })

    all_histories[label] = histories

    avg_gen = f"{np.mean(gens_to_solve):.1f}" if gens_to_solve else "N/A"
    print(f"  {label:<10} | success={successes}/{N_RUNS} | "
          f"avg_fit={np.mean(final_fits):.2f} | "
          f"avg_gen_solved={avg_gen}")

# ── Save raw results ──────────────────────────────────────────────────────────

with open("../results/raw_runs.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=raw_results[0].keys())
    writer.writeheader()
    writer.writerows(raw_results)

# ── Save summary ──────────────────────────────────────────────────────────────

summary = []
for exp in experiments:
    label = exp["label"]
    runs  = [r for r in raw_results if r["config"] == label]
    fits  = [r["best_fit"] for r in runs]
    gens  = [r["gen_found"] for r in runs if r["solved"] == 1]
    times = [r["runtime_s"] for r in runs]

    summary.append({
        "config":           label,
        "pop_size":         exp["pop_size"],
        "mutation_rate":    exp["mutation_rate"],
        "tournament_size":  exp["tournament_size"],
        "success_rate":     f"{sum(r['solved'] for r in runs)}/{N_RUNS}",
        "mean_fitness":     round(float(np.mean(fits)), 3),
        "std_fitness":      round(float(np.std(fits)), 3),
        "avg_gen_solved":   round(float(np.mean(gens)), 1) if gens else "N/A",
        "mean_time_ms":     round(float(np.mean(times)) * 1000, 1),
    })

with open("../results/summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

# ── Save best solutions ───────────────────────────────────────────────────────

with open("../results/best_solutions.txt", "w") as f:
    f.write("Best solutions found per configuration\n")
    f.write("=" * 45 + "\n\n")
    for exp in experiments:
        label  = exp["label"]
        runs   = [r for r in raw_results if r["config"] == label]
        best   = max(runs, key=lambda r: r["best_fit"])
        f.write(f"{label}: best_fit={best['best_fit']}, "
                f"gen={best['gen_found']}, run={best['run']}\n")

np.save("../results/histories.npy", all_histories, allow_pickle=True)

print("\nResults saved to ../results/")
