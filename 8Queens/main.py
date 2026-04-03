"""
main.py — Entry point for the 8-Queens Evolutionary Algorithm Project.

Runs in order:
    1. experiment.py  — parameter sweep, 30 trials per config
    2. plots.py       — all figures
    3. Prints final summary table
"""

import subprocess
import sys
import os
import csv

def run_script(name):
    print(f"\n{'='*55}")
    print(f"  Running {name} ...")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, name],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"\nError in {name}. Stopping.")
        sys.exit(1)

if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../figures", exist_ok=True)

    run_script("experiment.py")
    run_script("plots.py")

    print("\n\n" + "="*75)
    print("FINAL SUMMARY")
    print("="*75)

    with open("../results/summary.csv") as f:
        rows = list(csv.DictReader(f))

    header = (f"{'Config':<12} {'Pop':>5} {'Mut':>6} {'Tour':>5} "
              f"{'Success':>9} {'Mean Fit':>9} {'Std':>6} {'Avg Gen':>9} {'Time(ms)':>9}")
    print(header)
    print("-" * 75)
    for r in rows:
        print(f"{r['config']:<12} {r['pop_size']:>5} {r['mutation_rate']:>6} "
              f"{r['tournament_size']:>5} {r['success_rate']:>9} "
              f"{r['mean_fitness']:>9} {r['std_fitness']:>6} "
              f"{r['avg_gen_solved']:>9} {r['mean_time_ms']:>9}")

    print("\nAll outputs saved to results/ and figures/")
