# Project 1: Incremental Evolutionary Algorithm for 8-Queens

## Objective
Solve the 8-queens problem using an evolutionary algorithm with an incremental
permutation encoding. Queens are placed one per column left-to-right, using
unique row assignments (search space = 8! = 40,320). Compare performance across
population sizes, mutation rates, and tournament sizes.

## Representation
| Property | Detail |
|----------|--------|
| Encoding | Permutation of rows [0..7], one per column |
| Search space | 8! = 40,320 (row-distinct incremental) |
| Fitness | 28 - penalty, where penalty = attacking queen pairs |
| Perfect fitness | 28 (all C(8,2) pairs non-attacking) |

## Files
| File | Description |
|------|-------------|
| `algorithms.py` | Fitness, crossover, mutation, and the full EA loop |
| `experiment.py` | Parameter sweep for population size, mutation rate, and tournament size |
| `plots.py` | Board diagrams, convergence, success rates, boxplot, and search space plots |
| `main.py` | Single entry point |

## How to Run

```bash
cd 8Queens
python main.py
```

Or step by step:
```bash
cd 8Queens
python experiment.py
python plots.py
```

## Parameters Tested

| Sweep | Values | Fixed params |
|-------|--------|--------------|
| Population size | 20, 50, 100 | mut = 0.10, tour = 3 |
| Mutation rate | 0.05, 0.10, 0.20 | pop = 100, tour = 3 |
| Tournament size | 2, 3, 5 | pop = 100, mut = 0.10 |

Fixed across all: elitism = 2, crossover = cut-and-crossfill, max_gens = 1000, runs = 30

## Outputs

| Location | Contents |
|----------|----------|
| `results/raw_runs.csv` | Per-run fitness, solved flag, generation found, and runtime |
| `results/summary.csv` | Per-config success rate, mean fitness, and average generation to solve |
| `results/best_solutions.txt` | Best individual per configuration |
| `figures/example_boards.png` | Visual board with and without conflicts |
| `figures/search_space.png` | Search space size comparison |
| `figures/convergence_plot.png` | Mean plus/minus std fitness per generation |
| `figures/success_rates.png` | Success rate bar chart per configuration |
| `figures/boxplot.png` | Final fitness distribution across runs |

## Conclusion
The incremental permutation encoding dramatically reduces the search space from
C(64,8) = 4.4B to 8! = 40,320. With population = 100, mutation = 0.10, and
tournament size = 3, the EA reliably finds perfect solutions within a few hundred
generations.
