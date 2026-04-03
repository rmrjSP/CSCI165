import numpy as np
import os

RANDOM_SEED  = 42
N_CITIES     = 100
COORD_RANGE  = (0, 100)


def generate_cities(n=N_CITIES, seed=RANDOM_SEED):
    """
    Generate a fixed random 100-city instance.
    Saved to disk so the same dataset is reused across all algorithms.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(COORD_RANGE[0], COORD_RANGE[1], size=(n, 2))
    return coords


def load_or_create_cities(path="../results/cities.npy"):
    """Load city coordinates from disk, or create + save them if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return np.load(path)
    coords = generate_cities()
    np.save(path, coords)
    return coords


def make_rng_list(n_runs, base_seed=RANDOM_SEED):
    """Return a list of n_runs independent numpy RNG objects (reproducible)."""
    return [np.random.default_rng(base_seed + i) for i in range(n_runs)]
