# Project 3: Gradient Descent vs Simulated Annealing on the Rastrigin Function

## Objective
Minimize the 2D Rastrigin function using three variants of gradient descent
(fixed, decaying, momentum) and simulated annealing, then compare their
ability to find the global minimum.

## Files
| File | Description |
|------|-------------|
| `code/algorithms.py` | Rastrigin function, gradient, and all algorithm implementations |
| `code/experiment.py` | Runs 30 repeated trials per algorithm, saves CSV results |
| `code/plots.py`      | Generates all figures (surface, contour, trajectories, convergence, boxplot) |
| `code/main.py`       | Single entry point — runs everything in order |

## How to Run

### Run everything at once
```bash
cd code
python main.py
```

### Or run step by step
```bash
cd code
python experiment.py   # run trials → saves to results/
python plots.py        # generate figures → saves to figures/
```

## Parameters

| Algorithm | Parameter | Value |
|-----------|-----------|-------|
| GD Fixed | learning rate α | 0.01 |
| GD Decaying | initial α₀ | 0.1, decay = 0.001 |
| GD Momentum | α = 0.01, β (momentum) | 0.9 |
| SA | T₀ = 10, cooling α | 0.995, step radius = 0.5 |
| All | Max iterations | 5000 |
| All | Runs | 30 |

## Outputs

| Location | Contents |
|----------|----------|
| `results/raw_runs.csv` | Per-run data: best f, coordinates, success, runtime |
| `results/summary.csv` | Per-algorithm stats: mean, std, best, success rate, time |
| `results/best_solutions.txt` | Best point found per algorithm |
| `figures/rastrigin_3d.png` | 3D surface plot |
| `figures/rastrigin_contour.png` | Contour plot with global minimum marked |
| `figures/trajectories.png` | Search paths of each algorithm on contour map |
| `figures/convergence_plot.png` | Mean ± std convergence curves over 30 runs |
| `figures/boxplot.png` | Distribution of final best f values |

## Conclusion
Gradient descent is fast but sensitive to local minima in the Rastrigin landscape.
Simulated annealing explores more broadly via probabilistic acceptance of worse moves,
making it more robust at finding the global minimum f(0,0) = 0.
