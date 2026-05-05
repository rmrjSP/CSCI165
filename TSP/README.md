# Project 2: TSP with Hill Climbing, Simulated Annealing, and Threshold Accepting

## Objective
Compare three local-search algorithms on a fixed 100-city TSP instance:
Hill Climbing (plain + random restarts), Simulated Annealing (two cooling rates),
and Threshold Accepting. Determine which finds the shortest tour and why.

## Files
| File | Description |
|------|-------------|
| `algorithms.py` | TSP class, neighborhood operators, and all three algorithm implementations |
| `utils.py` | City dataset generation and RNG utilities |
| `experiment.py` | Runs 30 repeated trials per algorithm and saves CSV results |
| `plots.py` | Generates the city map, best routes, convergence, boxplot, and temperature schedule |
| `main.py` | Single entry point that runs everything in order |

## How to Run

### Run everything at once
```bash
cd TSP
python main.py
```

### Or step by step
```bash
cd TSP
python experiment.py
python plots.py
```

## Parameters

| Algorithm | Parameter | Value |
|-----------|-----------|-------|
| Hill Climbing | restarts | 1 (plain) or 5 (restart) |
| SA fast | T0 = 1000, cooling alpha | 0.999 |
| SA slow | T0 = 1000, cooling alpha | 0.9999 |
| Threshold Accepting | initial threshold | 50.0, decay = 0.995 |
| All | Neighbor operator | 2-opt reversal |
| All | Evaluation budget | 100,000 |
| All | Runs | 30 |

## Outputs

| Location | Contents |
|----------|----------|
| `results/cities.npy` | Fixed 100-city coordinate dataset |
| `results/raw_runs.csv` | Per-run results: best cost and runtime |
| `results/summary.csv` | Per-algorithm stats: mean, std, best, worst, and time |
| `results/best_solutions.txt` | Best route found per algorithm |
| `figures/city_map.png` | Scatter plot of all 100 cities |
| `figures/best_routes.png` | Best route visualization per algorithm |
| `figures/convergence_plot.png` | Mean plus/minus std convergence over 30 runs |
| `figures/boxplot.png` | Distribution of final tour costs |
| `figures/temperature_schedule.png` | SA cooling schedule comparison |

## Conclusion
Hill climbing converges quickly but gets trapped in local minima.
Simulated annealing with a slow cooling schedule typically finds the shortest tours
by escaping local minima via probabilistic acceptance of worse moves.
Threshold accepting sits between the two: deterministic but still explorative.
