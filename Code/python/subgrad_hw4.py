import numpy as np

np.random.seed(123)

n = 500
m = 100
x = np.zeros((n, 1))
A = np.random.randn(m, n)
lmbda = 10**-3
maxiter = 200

objective_values = []  # Store objective values for best selection

for k in range(1, maxiter + 1):
    # Calculate subgradient
    z = -A @ x
    exp_z = np.exp(z)
    grad_g = -A.T @ (exp_z / (1 + exp_z))  # Gradient of g(x)

    # Subgradient of lambda * ||x||_1
    subgrad_l1 = lmbda * np.sign(x)
    subgrad_l1[x == 0] = 0  # Handle non-differentiability at zero

    subgrad = grad_g + subgrad_l1

    # Diminishing stepsize
    alpha = 1 / np.sqrt(k)

    # Update x
    x = x - alpha * subgrad

    # Calculate objective value
    objective_value = np.sum(np.log(1 + np.exp(-A @ x))) + lmbda * np.linalg.norm(x, 1)
    objective_values.append(objective_value)

# Best objective value (min)
best_objective_value = min(objective_values)

# Count the number of zeros in the final x
num_zeros = np.sum(x == 0)

print(f'Best objective value from Subgradient Method: {best_objective_value}')
