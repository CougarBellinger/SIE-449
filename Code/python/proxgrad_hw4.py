import numpy as np

np.random.seed(123)

n = 500
m = 100
x = np.zeros((n, 1))
A = np.random.randn(m, n)
lmbda = 10**-3
maxiter = 200

L = np.linalg.norm(A.T @ A) / 4  # Lipschitz constant
alpha = 1 / L  # Fixed stepsize

for k in range(1, maxiter + 1):
    # Calculate gradient of g(x)
    z = -A @ x
    exp_z = np.exp(z)
    grad_g = -A.T @ (exp_z / (1 + exp_z))

    # Proximal gradient update
    x_new = x - alpha * grad_g
    x = np.maximum(np.abs(x_new) - alpha * lmbda, 0) * np.sign(x_new)

# Calculate the objective value of the last iterate
z = -A @ x
objective_value = np.sum(np.log(1 + np.exp(z))) + lmbda * np.linalg.norm(x, 1)

# Count the number of zeros in the final x
num_zeros = np.sum(x == 0)

print(f'Objective value from Proximal Gradient: {objective_value}')