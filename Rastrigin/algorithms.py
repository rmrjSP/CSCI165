import numpy as np

# ── Rastrigin function ────────────────────────────────────────────────────────
# 2D version: f(x1,x2) = 20 + x1² + x2² - 10(cos(2πx1) + cos(2πx2))
# Global minimum: f(0, 0) = 0
# Domain: x1, x2 ∈ [-5.12, 5.12]

DOMAIN_MIN = -5.12
DOMAIN_MAX =  5.12


def rastrigin(x):
    """Evaluate the 2D Rastrigin function at point x = [x1, x2]."""
    x1, x2 = x
    return 20 + x1**2 + x2**2 - 10 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))


def rastrigin_grad(x):
    """Analytical gradient of the 2D Rastrigin function."""
    x1, x2 = x
    g1 = 2 * x1 + 20 * np.pi * np.sin(2 * np.pi * x1)
    g2 = 2 * x2 + 20 * np.pi * np.sin(2 * np.pi * x2)
    return np.array([g1, g2])


def clip(x):
    """Clip a point to the valid domain [-5.12, 5.12]."""
    return np.clip(x, DOMAIN_MIN, DOMAIN_MAX)


# ── Gradient Descent Variants ─────────────────────────────────────────────────

def gradient_descent_fixed(x0, alpha=0.01, max_iter=5000, tol=1e-6):
    """
    Gradient descent with a fixed learning rate.

    Args:
        x0:       starting point [x1, x2]
        alpha:    fixed learning rate (step size)
        max_iter: maximum number of iterations
        tol:      stop early if gradient norm falls below this

    Returns:
        best_x:   point with lowest function value found
        best_f:   function value at best_x
        history:  list of f(x) values at each iteration (for convergence plot)
    """
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_f = rastrigin(x)
    history = [best_f]

    for _ in range(max_iter):
        grad = rastrigin_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        x = clip(x - alpha * grad)
        f = rastrigin(x)
        history.append(f)
        if f < best_f:
            best_f = f
            best_x = x.copy()

    return best_x, best_f, history


def gradient_descent_decaying(x0, alpha0=0.1, decay=0.001, max_iter=5000, tol=1e-6):
    """
    Gradient descent with a decaying learning rate: alpha_t = alpha0 / (1 + decay * t)

    Args:
        x0:       starting point
        alpha0:   initial learning rate
        decay:    controls how fast the rate decreases
        max_iter: maximum iterations
        tol:      early-stop gradient norm threshold

    Returns: same as gradient_descent_fixed
    """
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_f = rastrigin(x)
    history = [best_f]

    for t in range(max_iter):
        grad = rastrigin_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        alpha_t = alpha0 / (1 + decay * t)   # learning rate shrinks over time
        x = clip(x - alpha_t * grad)
        f = rastrigin(x)
        history.append(f)
        if f < best_f:
            best_f = f
            best_x = x.copy()

    return best_x, best_f, history


def gradient_descent_momentum(x0, alpha=0.01, beta=0.9, max_iter=5000, tol=1e-6):
    """
    Gradient descent with momentum.
    velocity_t = beta * velocity_{t-1} + alpha * grad_t
    x_{t+1}   = x_t - velocity_t

    Args:
        x0:       starting point
        alpha:    learning rate
        beta:     momentum coefficient (0 = no momentum, ~0.9 is typical)
        max_iter: maximum iterations
        tol:      early-stop gradient norm threshold

    Returns: same as gradient_descent_fixed
    """
    x = np.array(x0, dtype=float)
    velocity = np.zeros(2)
    best_x = x.copy()
    best_f = rastrigin(x)
    history = [best_f]

    for _ in range(max_iter):
        grad = rastrigin_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        velocity = beta * velocity + alpha * grad   # accumulate momentum
        x = clip(x - velocity)
        f = rastrigin(x)
        history.append(f)
        if f < best_f:
            best_f = f
            best_x = x.copy()

    return best_x, best_f, history


# ── Simulated Annealing ───────────────────────────────────────────────────────

def simulated_annealing(x0, T0=10.0, alpha=0.995, T_min=1e-4,
                        max_iter=5000, step_radius=0.5):
    """
    Simulated annealing for continuous minimization on Rastrigin.

    Neighbor: x_new = x + noise  where noise ~ Uniform(-step_radius, step_radius)
    Acceptance: always accept if better; accept worse with prob exp(-delta/T)

    Args:
        x0:          starting point
        T0:          initial temperature
        alpha:       cooling rate (T = alpha * T each step, exponential cooling)
        T_min:       stop when temperature drops below this
        max_iter:    hard cap on iterations
        step_radius: half-width of uniform perturbation

    Returns:
        best_x:   best point found
        best_f:   best function value
        history:  f(best) recorded at each iteration
    """
    x = np.array(x0, dtype=float)
    current_f = rastrigin(x)
    best_x = x.copy()
    best_f = current_f
    history = [best_f]
    T = T0

    for _ in range(max_iter):
        if T < T_min:
            break

        # Generate neighbor by small random perturbation
        noise = np.random.uniform(-step_radius, step_radius, size=2)
        x_new = clip(x + noise)
        new_f = rastrigin(x_new)

        delta = new_f - current_f

        # Accept if better, or probabilistically if worse
        if delta <= 0 or np.random.rand() < np.exp(-delta / T):
            x = x_new
            current_f = new_f

        if current_f < best_f:
            best_f = current_f
            best_x = x.copy()

        history.append(best_f)
        T *= alpha   # cool down

    return best_x, best_f, history
