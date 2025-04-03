import numpy as np

def projected_gradient(A, b, epsilon=1e-3, max_iters=1000):
    """
    Solves the quadratic optimization problem using the projected gradient method.

    Args:
        A: The symmetric positive definite matrix A.
        b: The vector b.
        epsilon: The stopping criterion for the algorithm.
        max_iters: The maximum number of iterations.

    Returns:
        x: The optimal solution.
        f_val: The function value at the optimal solution.
    """

    n = A.shape[0]
    x = np.zeros(n)  # Initial point x0 = 0
    L = 2 * np.linalg.norm(A, 2)  # Lipschitz constant of the gradient
    alpha = 1 / L  # Step size

    def objective_function(x):
        return 2 * x.T @ A @ x + b.T @ x

    def projection(x):
        """Projects x onto the feasible set {x | sum(x_i) >= -10}."""
        if np.sum(x) >= -10:
            return x
        else:
            # Find the closest point in the feasible set
            diff = -10 - np.sum(x)
            return x + diff / n

    for _ in range(max_iters):
        grad = 2* A @ x + b
        x_new = projection(x - alpha * grad)
        if np.linalg.norm(x_new - x) <= epsilon:
            break
        x = x_new

    f_val = objective_function(x)
    return x, f_val

# Set the seed for reproducibility
np.random.seed(123)

# Generate the problem parameters
n = 10
R = np.random.rand(n, n)
A = R.T @ R
b = np.random.rand(n, 1)

# Solve the problem using the projected gradient method
x_opt, f_val_opt = projected_gradient(A, b)

# Display the function value at the last iteration point
print("Function value at the last iteration point:", f_val_opt[0, 0])