"""
main.py — Entry point for the Rastrigin Optimization Project.

Run order:
    1. Generates all figures (surface, contour, trajectories)
    2. Runs all experiments (30 trials per algorithm)
    3. Generates results-dependent figures (convergence, boxplot)
    4. Prints final summary table
"""

import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print('='*50)
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"Error in {script_name}. Stopping.")
        sys.exit(1)

if __name__ == "__main__":
    # Make sure output folders exist
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../figures", exist_ok=True)

    run_script("experiment.py")
    run_script("plots.py")

    # Print the summary table
    print("\n\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    import csv
    with open("../results/summary.csv") as f:
        rows = list(csv.DictReader(f))

    header = f"{'Algorithm':<15} {'Mean f':>10} {'Std f':>10} {'Best f':>10} {'Success':>10} {'Time(ms)':>10}"
    print(header)
    print("-" * 65)
    for r in rows:
        print(f"{r['algorithm']:<15} {r['mean_f']:>10} {r['std_f']:>10} {r['best_f']:>10} "
              f"{r['success_rate']:>10} {r['mean_time_ms']:>10}")

    print("\nAll results saved to results/ and figures/")
