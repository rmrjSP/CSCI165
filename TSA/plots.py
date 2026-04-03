import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import os

os.makedirs("../figures", exist_ok=True)

COLORS = {
    "HC_plain":   "#2196F3",
    "HC_restart": "#03A9F4",
    "SA_fast":    "#F44336",
    "SA_slow":    "#FF9800",
    "TA":         "#4CAF50",
}

ALGO_LABELS = {
    "HC_plain":   "HC (plain)",
    "HC_restart": "HC (5 restarts)",
    "SA_fast":    "SA (α=0.999)",
    "SA_slow":    "SA (α=0.9999)",
    "TA":         "Threshold Accepting",
}


def load_data():
    histories  = np.load("../results/histories.npy",   allow_pickle=True).item()
    best_routes = np.load("../results/best_routes.npy", allow_pickle=True).item()
    coords     = np.load("../results/cities.npy")
    with open("../results/raw_runs.csv") as f:
        raw = list(csv.DictReader(f))
    return histories, best_routes, coords, raw


# ── 1. City scatter plot ──────────────────────────────────────────────────────

def plot_cities(coords):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=25, color="#555", zorder=3)
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), fontsize=4, alpha=0.6,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_title("100-City TSP Instance")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    plt.tight_layout()
    plt.savefig("../figures/city_map.png", dpi=150)
    plt.close()
    print("Saved: city_map.png")


# ── 2. Best route visualizations (one panel per algorithm) ────────────────────

def plot_best_routes(coords, best_routes):
    algo_names = list(best_routes.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, name in zip(axes, algo_names):
        route, cost = best_routes[name]
        route = list(route) + [route[0]]   # close the loop
        xs = coords[route, 0]
        ys = coords[route, 1]
        color = COLORS[name]

        ax.plot(xs, ys, '-', color=color, lw=0.9, alpha=0.7)
        ax.scatter(coords[:, 0], coords[:, 1], s=12, color="#333", zorder=4)
        ax.plot(coords[route[0], 0], coords[route[0], 1],
                'r*', markersize=10, zorder=5, label="Start")
        ax.set_title(f"{ALGO_LABELS[name]}\ncost = {cost:.2f}", fontsize=10)
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
        ax.axis("off")

    # Hide unused subplot (we have 5 algos, 6 panels)
    axes[-1].axis("off")
    plt.suptitle("Best Route Found per Algorithm", fontsize=13)
    plt.tight_layout()
    plt.savefig("../figures/best_routes.png", dpi=150)
    plt.close()
    print("Saved: best_routes.png")


# ── 3. Convergence curves (mean ± std) ───────────────────────────────────────

def plot_convergence(histories):
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, runs in histories.items():
        max_len = max(len(h) for h in runs)
        padded  = np.array([h + [h[-1]] * (max_len - len(h)) for h in runs])
        mean    = padded.mean(axis=0)
        std     = padded.std(axis=0)
        x       = np.arange(max_len)

        ax.plot(x, mean, label=ALGO_LABELS[name], color=COLORS[name], lw=1.8)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[name], alpha=0.12)

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best tour cost")
    ax.set_title("Convergence Curves (mean ± std over 30 runs)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("../figures/convergence_plot.png", dpi=150)
    plt.close()
    print("Saved: convergence_plot.png")


# ── 4. Boxplot of final tour costs ───────────────────────────────────────────

def plot_boxplot(raw):
    algo_names = list(COLORS.keys())
    data   = [[float(r["best_cost"]) for r in raw if r["algorithm"] == n]
               for n in algo_names]
    labels = [ALGO_LABELS[n] for n in algo_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5)

    for patch, name in zip(bp["boxes"], algo_names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.72)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Final best tour cost")
    ax.set_title("Distribution of Final Tour Costs Across 30 Runs")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("../figures/boxplot.png", dpi=150)
    plt.close()
    print("Saved: boxplot.png")


# ── 5. SA temperature schedule illustration ──────────────────────────────────

def plot_temperature_schedules():
    iters = np.arange(100_000)
    T_fast = 1000.0 * (0.9990 ** iters)
    T_slow = 1000.0 * (0.9999 ** iters)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, T_fast, color="#F44336", lw=2, label="SA fast (α=0.999)")
    ax.plot(iters, T_slow, color="#FF9800", lw=2, label="SA slow (α=0.9999)")
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Temperature T")
    ax.set_title("SA Cooling Schedules")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("../figures/temperature_schedule.png", dpi=150)
    plt.close()
    print("Saved: temperature_schedule.png")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...\n")

    try:
        histories, best_routes, coords, raw = load_data()
        plot_cities(coords)
        plot_best_routes(coords, best_routes)
        plot_convergence(histories)
        plot_boxplot(raw)
        plot_temperature_schedules()
    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        print("Run experiment.py first, then re-run plots.py")

    print("\nAll figures saved to ../figures/")
