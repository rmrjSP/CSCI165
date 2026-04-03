import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os

os.makedirs("../figures", exist_ok=True)

# Color palette per parameter sweep group
SWEEP_COLORS = {
    "pop":  ["#90CAF9", "#2196F3", "#0D47A1"],   # blues  → pop 20, 50, 100
    "mut":  ["#A5D6A7", "#4CAF50", "#1B5E20"],   # greens → mut 0.05, 0.10, 0.20
    "tour": ["#FFCC80", "#FF9800", "#E65100"],   # oranges→ tour 2, 3, 5
}

SWEEP_GROUPS = {
    "pop":  ["pop=20",   "pop=50",   "pop=100"],
    "mut":  ["mut=0.05", "mut=0.10", "mut=0.20"],
    "tour": ["tour=2",   "tour=3",   "tour=5"],
}


def load_data():
    histories = np.load("../results/histories.npy", allow_pickle=True).item()
    with open("../results/raw_runs.csv") as f:
        raw = list(csv.DictReader(f))
    with open("../results/summary.csv") as f:
        summary = list(csv.DictReader(f))
    return histories, raw, summary


# ── 1. Board visualisation ────────────────────────────────────────────────────

def draw_board(ax, individual, title=""):
    """Draw an 8x8 chessboard with queens placed according to individual."""
    n = len(individual)
    for row in range(n):
        for col in range(n):
            color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
            ax.add_patch(patches.Rectangle((col, row), 1, 1, color=color))

    for col, row in enumerate(individual):
        ax.text(col + 0.5, row + 0.5, "♛",
                ha="center", va="center", fontsize=18,
                color="#1a1a1a" if (row + col) % 2 == 0 else "#f5f5f5")

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)


def plot_example_boards():
    """Show a solved board and a partial-conflict board side by side."""
    from algorithms import evolutionary_algorithm, fitness, MAX_PAIRS
    rng = np.random.default_rng(0)

    # Find a perfect solution
    perfect = None
    for seed in range(200):
        ind, fit, _, _ = evolutionary_algorithm(pop_size=100, max_gens=500,
                                                rng=np.random.default_rng(seed))
        if fit == MAX_PAIRS:
            perfect = ind
            break

    # Generate a random (likely imperfect) board
    imperfect = np.array([0, 4, 7, 5, 2, 6, 1, 3])  # known partial conflict
    fit_imp   = fitness(imperfect)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    draw_board(axes[0], imperfect,
               f"Example Board  (fitness = {fit_imp}/28)")
    if perfect is not None:
        draw_board(axes[1], perfect,
                   f"Perfect Solution (fitness = 28/28)")
    else:
        axes[1].text(0.5, 0.5, "No perfect\nsolution found",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.suptitle("8-Queens Board Representations", fontsize=13)
    plt.tight_layout()
    plt.savefig("../figures/example_boards.png", dpi=150)
    plt.close()
    print("Saved: example_boards.png")


# ── 2. Convergence curves (one panel per sweep) ───────────────────────────────

def plot_convergence(histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (sweep_name, configs) in zip(axes, SWEEP_GROUPS.items()):
        colors = SWEEP_COLORS[sweep_name]
        for config, color in zip(configs, colors):
            runs    = histories[config]
            max_len = max(len(h) for h in runs)
            padded  = np.array([h + [h[-1]] * (max_len - len(h)) for h in runs])
            mean    = padded.mean(axis=0)
            std     = padded.std(axis=0)
            x       = np.arange(max_len)
            ax.plot(x, mean, label=config, color=color, lw=2)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        ax.axhline(y=28, color="red", linestyle="--", alpha=0.5, label="Perfect (28)")
        ax.set_xlabel("Generation")
        ax.set_title(f"{sweep_name.capitalize()} sweep")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylim(18, 29)

    axes[0].set_ylabel("Best fitness")
    plt.suptitle("Convergence Curves (mean ± std over 30 runs)", fontsize=13)
    plt.tight_layout()
    plt.savefig("../figures/convergence_plot.png", dpi=150)
    plt.close()
    print("Saved: convergence_plot.png")


# ── 3. Success rate bar chart ─────────────────────────────────────────────────

def plot_success_rates(summary):
    labels  = [r["config"] for r in summary]
    # Parse "X/30" → float
    rates   = [int(r["success_rate"].split("/")[0]) / 30 * 100 for r in summary]

    # Colour bars by sweep group
    colors = []
    for lbl in labels:
        if lbl.startswith("pop"):
            idx = ["pop=20", "pop=50", "pop=100"].index(lbl)
            colors.append(SWEEP_COLORS["pop"][idx])
        elif lbl.startswith("mut"):
            idx = ["mut=0.05", "mut=0.10", "mut=0.20"].index(lbl)
            colors.append(SWEEP_COLORS["mut"][idx])
        else:
            idx = ["tour=2", "tour=3", "tour=5"].index(lbl)
            colors.append(SWEEP_COLORS["tour"][idx])

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.8)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{rate:.0f}%",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Success Rate per Configuration (30 runs)")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)

    # Legend separators
    from matplotlib.patches import Patch
    legend = [
        Patch(color=SWEEP_COLORS["pop"][1],  label="Population sweep"),
        Patch(color=SWEEP_COLORS["mut"][1],  label="Mutation sweep"),
        Patch(color=SWEEP_COLORS["tour"][1], label="Tournament sweep"),
    ]
    ax.legend(handles=legend, fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig("../figures/success_rates.png", dpi=150)
    plt.close()
    print("Saved: success_rates.png")


# ── 4. Boxplot of final fitness ───────────────────────────────────────────────

def plot_boxplot(raw):
    configs = [e["label"] for e in [
        {"label": "pop=20"},  {"label": "pop=50"},  {"label": "pop=100"},
        {"label": "mut=0.05"},{"label": "mut=0.10"},{"label": "mut=0.20"},
        {"label": "tour=2"},  {"label": "tour=3"},  {"label": "tour=5"},
    ]]
    data = [[int(r["best_fit"]) for r in raw if r["config"] == c] for c in configs]

    all_colors = (SWEEP_COLORS["pop"] + SWEEP_COLORS["mut"] + SWEEP_COLORS["tour"])

    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.55)
    for patch, color in zip(bp["boxes"], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Best fitness (max=28)")
    ax.set_title("Distribution of Best Fitness Across 30 Runs")
    ax.axhline(y=28, color="red", linestyle="--", alpha=0.5, label="Perfect solution")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("../figures/boxplot.png", dpi=150)
    plt.close()
    print("Saved: boxplot.png")


# ── 5. Search space illustration ─────────────────────────────────────────────

def plot_search_space():
    """Bar chart comparing search space sizes."""
    labels = ["All boards\nC(64,8)", "Unrestricted\nincremental\n8^8",
              "Permutation\nincremental\n8!"]
    sizes  = [4_426_165_368, 16_777_216, 40_320]
    colors = ["#EF9A9A", "#FFF176", "#A5D6A7"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, sizes, color=colors, edgecolor="#555", linewidth=0.8)
    for bar, sz in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{sz:,}", ha="center", va="bottom", fontsize=9)

    ax.set_yscale("log")
    ax.set_ylabel("Search space size (log scale)")
    ax.set_title("Search Space Comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("../figures/search_space.png", dpi=150)
    plt.close()
    print("Saved: search_space.png")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...\n")
    plot_example_boards()
    plot_search_space()
    try:
        histories, raw, summary = load_data()
        plot_convergence(histories)
        plot_success_rates(summary)
        plot_boxplot(raw)
    except FileNotFoundError as e:
        print(f"Missing file: {e}\nRun experiment.py first.")
    print("\nAll figures saved to ../figures/")
