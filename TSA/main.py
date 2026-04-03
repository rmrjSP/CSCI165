"""
main.py — Entry point for the TSP Optimization Project.

Runs in order:
    1. experiment.py  — generates city instance, runs 30 trials per algorithm
    2. plots.py       — generates all figures
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

    # Print final summary table
    print("\n\n" + "="*65)
    print("FINAL SUMMARY")
    print("="*65)

    with open("../results/summary.csv") as f:
        rows = list(csv.DictReader(f))

    header = (f"{'Algorithm':<14} {'Mean Cost':>10} {'Std':>8} "
              f"{'Best':>10} {'Worst':>10} {'Time(ms)':>10}")
    print(header)
    print("-" * 65)
    for r in rows:
        print(f"{r['algorithm']:<14} {r['mean_cost']:>10} {r['std_cost']:>8} "
              f"{r['best_cost']:>10} {r['worst_cost']:>10} {r['mean_time_ms']:>10}")

    print("\nAll outputs saved to results/ and figures/")
