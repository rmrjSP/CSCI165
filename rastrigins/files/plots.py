import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from algorithms import rastrigin, rastrigin_grad, DOMAIN_MIN, DOMAIN_MAX

os.makedirs("../figures", exist_ok=True)

COLORS = {
    "GD_Fixed":    "#2196F3",
    "GD_Decaying": "#FF9800",
    "GD_Momentum": "#9C27B0",
    "SA":          "#F44336",
}

# ── 1. 3D Surface Plot ────────────────────────────────────────────────────────

def plot_3d_surface():
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 200)
    y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 200)
    X, Y = np.meshgrid(x, y)
    Z = 20 + X**2 + Y**2 - 10 * (np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, linewidth=0)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("f(x₁, x₂)")
    ax.set_title("Rastrigin Function – 3D Surface")
    fig.colorbar(surf, shrink=0.5, label="f value")
    plt.tight_layout()
    plt.savefig("../figures/rastrigin_3d.png", dpi=150)
    plt.close()
    print("Saved: rastrigin_3d.png")


# ── 2. Contour Plot ───────────────────────────────────────────────────────────

def plot_contour():
    x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 400)
    y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 400)
    X, Y = np.meshgrid(x, y)
    Z = 20 + X**2 + Y**2 - 10 * (np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))

    fig, ax = plt.subplots(figsize=(7, 6))
    cp = ax.contourf(X, Y, Z, levels=40, cmap='viridis')
    ax.contour(X, Y, Z, levels=40, colors='white', linewidths=0.3, alpha=0.3)
    plt.colorbar(cp, ax=ax, label="f(x₁, x₂)")
    ax.plot(0, 0, 'r*', markersize=14, label="Global min (0,0)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("Rastrigin Function – Contour Plot")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/rastrigin_contour.png", dpi=150)
    plt.close()
    print("Saved: rastrigin_contour.png")


# ── 3. Trajectory Plot ────────────────────────────────────────────────────────

def plot_trajectories():
    """Show how each algorithm moves through the landscape from the same start."""
    from algorithms import (
        gradient_descent_fixed, gradient_descent_decaying,
        gradient_descent_momentum, simulated_annealing
    )

    # Use a challenging starting point far from origin
    x0 = np.array([4.0, 4.0])

    def track_gd_fixed(x0):
        x = np.array(x0, dtype=float)
        path = [x.copy()]
        for _ in range(300):
            grad = rastrigin_grad(x)
            x = np.clip(x - 0.01 * grad, DOMAIN_MIN, DOMAIN_MAX)
            path.append(x.copy())
        return np.array(path)

    def track_gd_decaying(x0):
        x = np.array(x0, dtype=float)
        path = [x.copy()]
        for t in range(300):
            grad = rastrigin_grad(x)
            alpha_t = 0.1 / (1 + 0.001 * t)
            x = np.clip(x - alpha_t * grad, DOMAIN_MIN, DOMAIN_MAX)
            path.append(x.copy())
        return np.array(path)

    def track_gd_momentum(x0):
        x = np.array(x0, dtype=float)
        v = np.zeros(2)
        path = [x.copy()]
        for _ in range(300):
            grad = rastrigin_grad(x)
            v = 0.9 * v + 0.01 * grad
            x = np.clip(x - v, DOMAIN_MIN, DOMAIN_MAX)
            path.append(x.copy())
        return np.array(path)

    def track_sa(x0, seed=42):
        np.random.seed(seed)
        x = np.array(x0, dtype=float)
        T = 10.0
        path = [x.copy()]
        current_f = rastrigin(x)
        for _ in range(300):
            noise = np.random.uniform(-0.5, 0.5, 2)
            x_new = np.clip(x + noise, DOMAIN_MIN, DOMAIN_MAX)
            delta = rastrigin(x_new) - current_f
            if delta <= 0 or np.random.rand() < np.exp(-delta / T):
                x = x_new
                current_f = rastrigin(x)
            path.append(x.copy())
            T *= 0.995
        return np.array(path)

    paths = {
        "GD_Fixed":    track_gd_fixed(x0),
        "GD_Decaying": track_gd_decaying(x0),
        "GD_Momentum": track_gd_momentum(x0),
        "SA":          track_sa(x0),
    }

    # Background contour
    x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 300)
    y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 300)
    X, Y = np.meshgrid(x, y)
    Z = 20 + X**2 + Y**2 - 10*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (name, path) in zip(axes, paths.items()):
        ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
        ax.plot(path[:, 0], path[:, 1], '-', color=COLORS[name], lw=1.5, alpha=0.8)
        ax.plot(path[0, 0], path[0, 1], 'o', color='white', markersize=7, label="Start")
        ax.plot(path[-1, 0], path[-1, 1], 's', color=COLORS[name], markersize=7, label="End")
        ax.plot(0, 0, 'r*', markersize=10, label="Global min")
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.legend(fontsize=8)

    plt.suptitle("Algorithm Trajectories from Start (4.0, 4.0)", fontsize=13)
    plt.tight_layout()
    plt.savefig("../figures/trajectories.png", dpi=150)
    plt.close()
    print("Saved: trajectories.png")


# ── 4. Convergence Curves ─────────────────────────────────────────────────────

def plot_convergence():
    histories = np.load("../results/histories.npy", allow_pickle=True).item()

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, runs in histories.items():
        # Pad shorter histories to the same length for averaging
        max_len = max(len(h) for h in runs)
        padded = [h + [h[-1]] * (max_len - len(h)) for h in runs]
        arr = np.array(padded)
        mean_curve = arr.mean(axis=0)
        std_curve  = arr.std(axis=0)
        x_axis = np.arange(max_len)
        ax.plot(x_axis, mean_curve, label=name, color=COLORS[name], lw=2)
        ax.fill_between(x_axis,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=COLORS[name], alpha=0.15)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best f(x) found")
    ax.set_title("Convergence Curves (mean ± std over 30 runs)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("../figures/convergence_plot.png", dpi=150)
    plt.close()
    print("Saved: convergence_plot.png")


# ── 5. Boxplot ────────────────────────────────────────────────────────────────

def plot_boxplot():
    import csv
    with open("../results/raw_runs.csv") as f:
        rows = list(csv.DictReader(f))

    algo_names = ["GD_Fixed", "GD_Decaying", "GD_Momentum", "SA"]
    data = [[float(r["best_f"]) for r in rows if r["algorithm"] == name]
            for name in algo_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, name in zip(bp['boxes'], algo_names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.7)

    ax.set_xticklabels(algo_names)
    ax.set_ylabel("Final best f(x)")
    ax.set_title("Distribution of Final Best Values Across 30 Runs")
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label="Global min (f=0)")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("../figures/boxplot.png", dpi=150)
    plt.close()
    print("Saved: boxplot.png")


# ── Run all plots ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...\n")
    plot_3d_surface()
    plot_contour()
    plot_trajectories()

    # These two need experiment.py to have run first
    try:
        plot_convergence()
        plot_boxplot()
    except FileNotFoundError:
        print("Note: Run experiment.py first to generate convergence + boxplot figures.")

    print("\nAll figures saved to ../figures/")
